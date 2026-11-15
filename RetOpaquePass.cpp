#include "RetOpaquePass.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

using namespace llvm;

llvm::cl::opt<bool> RetOpaqueEnable("obfretopq", llvm::cl::init(true), llvm::cl::desc("Return-address opaque dispatch pass."));

static constexpr unsigned FakeCountMin = 5;
static constexpr unsigned FakeCountMax = 9;
static constexpr unsigned DonorCloneCap = 6;
static constexpr unsigned DecoyMinTailOps = 2;
static constexpr unsigned DecoyMaxTailOps = 6;
static constexpr unsigned DecoyStoreChancePct = 10;

static bool HasAnnotationTag(Function& f, const char* tag)
{
    Module* m = f.getParent();
    GlobalVariable* ga = m->getGlobalVariable("llvm.global.annotations");
    if (!ga || !ga->hasInitializer()) return false;
    auto* ca = dyn_cast<ConstantArray>(ga->getOperand(0));
    if (!ca) return false;
    for (unsigned i = 0; i < ca->getNumOperands(); ++i)
    {
        auto* cs = dyn_cast<ConstantStruct>(ca->getOperand(i));
        if (!cs || cs->getNumOperands() < 2) continue;
        if (cs->getOperand(0)->stripPointerCasts() != &f) continue;
        auto* annogv = dyn_cast<GlobalVariable>(cs->getOperand(1)->stripPointerCasts());
        if (!annogv || !annogv->hasInitializer()) continue;
        auto* cds = dyn_cast<ConstantDataSequential>(annogv->getInitializer());
        if (!cds || !cds->isString()) continue;
        if (strcmp(cds->getAsString().str().c_str(), tag) == 0) return true;
    }
    return false;
}

static bool HasObfuscateAnnotation(Function& f)   { return HasAnnotationTag(f, "obfus"); }
static bool HasNoObfuscateAnnotation(Function& f) { return HasAnnotationTag(f, "noobfus"); }
static bool HasNoRetOpaqueAnnotation(Function& f) { return HasAnnotationTag(f, "noretopq"); }

static std::mt19937_64 Rng;

static bool FunctionAlreadyProcessed(Function& f)
{
    BasicBlock* bb = &f.getEntryBlock();
    Instruction* term = bb->getTerminator();
    return term && term->getMetadata("retopq.done") != nullptr;
}

static void MarkFunctionProcessed(Function& f)
{
    BasicBlock* bb = &f.getEntryBlock();
    Instruction* term = bb->getTerminator();
    if (!term) return;
    term->setMetadata("retopq.done", MDNode::get(bb->getContext(), {}));
}

static bool IsDecoyBlock(BasicBlock* bb)
{
    Instruction* term = bb->getTerminator();
    return term && term->getMetadata("retopq.decoy") != nullptr;
}

static void MarkDecoyBlock(BasicBlock* bb)
{
    Instruction* term = bb->getTerminator();
    if (!term) return;
    term->setMetadata("retopq.decoy", MDNode::get(bb->getContext(), {}));
}

static void TagEscalateConstMBA(Value* v)
{
    auto* inst = dyn_cast<Instruction>(v);
    if (!inst) return;
    LLVMContext& c = inst->getContext();
    Metadata* ops[1] = { MDString::get(c, "poly_const") };
    inst->setMetadata("mbagen.escalate", MDNode::get(c, ops));
}

// empty body inline asm with "=r,0" takes zero final binary instructions but makes LLVM treat its input as an opaque runtime value so constant folding cannot see through the identity
static Value* InsertFoldBarrier(IRBuilder<>& b, Value* v)
{
    Type* ty = v->getType();
    if (!ty->isIntegerTy()) return v;
    FunctionType* asmty = FunctionType::get(ty, { ty }, false);
    InlineAsm* ia = InlineAsm::get(asmty, "", "=r,0", true);
    return b.CreateCall(ia, { v }, "");
}

static GlobalVariable* GetOrCreateRetOpaqueSink(Module& m)
{
    const char* name = ".retopq.sink";
    if (auto* existing = m.getGlobalVariable(name)) return existing;
    LLVMContext& c = m.getContext();
    IntegerType* i64 = Type::getInt64Ty(c);
    auto* gv = new GlobalVariable(m, i64, false, GlobalValue::InternalLinkage, ConstantInt::get(i64, 0), name);
    gv->setUnnamedAddr(GlobalValue::UnnamedAddr::None);
    gv->setAlignment(Align(8));
    return gv;
}

static Value* EmitSinkLoad(IRBuilder<>& b, Module& m, IntegerType* i64, const char* name)
{
    GlobalVariable* sink = GetOrCreateRetOpaqueSink(m);
    LoadInst* ld = b.CreateLoad(i64, sink, true, name);
    ld->setAlignment(Align(8));
    return ld;
}

struct HashConsts
{
    uint64_t K0;
    uint64_t K1;
    uint64_t K2;
    uint64_t K3;
    uint64_t Kmix;
};

static HashConsts RollHashConsts(std::mt19937_64& rng)
{
    HashConsts h;
    h.K0 = ((uint64_t)rng()) | 1ull;
    h.K1 = ((uint64_t)rng()) | 1ull;
    h.K2 = ((uint64_t)rng()) | 1ull;
    h.K3 = ((uint64_t)rng()) | 1ull;
    h.Kmix = (uint64_t)rng();
    return h;
}

// both sides of the real match compare run EmitHashIR at runtime with the same HashConsts so hash(ra)==hash(ba) iff ra==ba; the FoldBarrier wraps on both inputs are what stop LLVM from recognising the bijection and folding the compare to ra==ba
static Value* EmitHashIR(IRBuilder<>& b, Value* xi64, IntegerType* i64, const HashConsts& hc)
{
    Value* r = InsertFoldBarrier(b, xi64);

    Value* s1 = b.CreateLShr(r, ConstantInt::get(i64, 30), "");
    r = b.CreateXor(r, s1, "");
    r = b.CreateMul(r, ConstantInt::get(i64, hc.K0), "");
    Value* s2 = b.CreateLShr(r, ConstantInt::get(i64, 27), "");
    r = b.CreateXor(r, s2, "");
    r = b.CreateMul(r, ConstantInt::get(i64, hc.K1), "");
    Value* s3 = b.CreateLShr(r, ConstantInt::get(i64, 31), "");
    r = b.CreateXor(r, s3, "");

    Value* kmix = InsertFoldBarrier(b, ConstantInt::get(i64, hc.Kmix));
    r = b.CreateXor(r, kmix, "");

    Value* s4 = b.CreateLShr(r, ConstantInt::get(i64, 29), "");
    r = b.CreateXor(r, s4, "");
    r = b.CreateMul(r, ConstantInt::get(i64, hc.K2), "");
    Value* s5 = b.CreateLShr(r, ConstantInt::get(i64, 25), "");
    r = b.CreateXor(r, s5, "");
    r = b.CreateMul(r, ConstantInt::get(i64, hc.K3), "");
    Value* s6 = b.CreateLShr(r, ConstantInt::get(i64, 32), "");
    r = b.CreateXor(r, s6, "");

    return r;
}

struct PolluteSlot
{
    AllocaInst* Base;
    int FieldIdx;
    Type* BaseTy;
    Type* RawTy;
    bool IsPtr;
    IntegerType* IntTy;
};

static Value* MaterializeSlotPtr(IRBuilder<>& b, const PolluteSlot& s)
{
    if (s.FieldIdx < 0) return s.Base;
    LLVMContext& c = b.getContext();
    IntegerType* i32 = Type::getInt32Ty(c);
    return b.CreateInBoundsGEP(s.BaseTy, s.Base, { ConstantInt::get(i32, 0), ConstantInt::get(i32, s.FieldIdx) }, "");
}

static Value* EmitSlotLoad(IRBuilder<>& b, const PolluteSlot& s)
{
    Value* ptr = MaterializeSlotPtr(b, s);
    LoadInst* ld = b.CreateLoad(s.RawTy, ptr, "");
    if (!s.IsPtr) return ld;
    return b.CreatePtrToInt(ld, s.IntTy, "");
}

static void EmitSlotStore(IRBuilder<>& b, const PolluteSlot& s, Value* intval, bool volatilestore)
{
    Value* ptr = MaterializeSlotPtr(b, s);
    Value* v = intval;
    if (s.IsPtr) v = b.CreateIntToPtr(intval, s.RawTy, "");
    StoreInst* st = b.CreateStore(v, ptr);
    if (volatilestore) st->setVolatile(true);
}

static void CollectPolluteSlots(Function& f, IntegerType* i32, IntegerType* i64, SmallVectorImpl<PolluteSlot>& out32, SmallVectorImpl<PolluteSlot>& out64)
{
    for (Instruction& i : f.getEntryBlock())
    {
        auto* a = dyn_cast<AllocaInst>(&i);
        if (!a || !a->isStaticAlloca()) continue;
        Type* ty = a->getAllocatedType();

        if (ty->isIntegerTy(32))
        {
            out32.push_back({a, -1, ty, ty, false, i32});
            continue;
        }
        if (ty->isIntegerTy(64))
        {
            out64.push_back({a, -1, ty, ty, false, i64});
            continue;
        }
        if (ty->isPointerTy())
        {
            out64.push_back({a, -1, ty, ty, true, i64});
            continue;
        }
        if (auto* st = dyn_cast<StructType>(ty))
        {
            for (unsigned fi = 0; fi < st->getNumElements(); ++fi)
            {
                Type* ft = st->getElementType(fi);
                if (ft->isIntegerTy(32)) out32.push_back({a, (int)fi, ty, ft, false, i32});
                else if (ft->isIntegerTy(64)) out64.push_back({a, (int)fi, ty, ft, false, i64});
                else if (ft->isPointerTy()) out64.push_back({a, (int)fi, ty, ft, true, i64});
            }
        }
    }
}

static bool CanCloneWithDominance(const Instruction& i, BasicBlock* switchsrc, DominatorTree& dt, const ValueToValueMapTy& vmap)
{
    for (const Use& u : i.operands())
    {
        Value* op = u.get();
        if (!op) continue;
        if (isa<Constant>(op)) continue;
        if (isa<Argument>(op)) continue;
        if (isa<BasicBlock>(op)) continue;
        if (isa<MetadataAsValue>(op)) continue;
        if (vmap.count(op)) continue;
        if (auto* oi = dyn_cast<Instruction>(op))
        {
            if (dt.dominates(oi, switchsrc->getTerminator())) continue;
            return false;
        }
        return false;
    }
    return true;
}

static unsigned DiamondCloneDonorInto(BasicBlock* donor, BasicBlock* decoy, BasicBlock* switchsrc, DominatorTree& dt, std::mt19937_64& rng, unsigned maxclone)
{
    if (maxclone == 0) return 0;
    if (!donor || !decoy) return 0;

    ValueToValueMapTy vmap;
    unsigned cloned = 0;
    const uint64_t mutnonce = rng();

    for (Instruction& i : *donor)
    {
        if (cloned >= maxclone) break;
        if (i.isTerminator()) break;

        if (isa<PHINode>(&i)) continue;
        if (isa<AllocaInst>(&i)) continue;
        if (isa<CallBase>(&i)) continue;
        if (isa<LandingPadInst>(&i)) continue;
        if (isa<CatchPadInst>(&i)) continue;
        if (isa<CleanupPadInst>(&i)) continue;
        if (i.isAtomic()) continue;
        if (isa<FenceInst>(&i)) continue;
        if (i.mayHaveSideEffects()) continue;

        if (!CanCloneWithDominance(i, switchsrc, dt, vmap)) continue;

        Instruction* clone = i.clone();
        clone->setName("");
        clone->insertInto(decoy, decoy->end());
        RemapInstruction(clone, vmap, RF_NoModuleLevelChanges | RF_IgnoreMissingLocals);

        // strip inherited escalate tags so the blanket skip below does not miss a clone that was pre tagged by an earlier pass
        clone->setMetadata("mbagen.escalate", nullptr);

        if (auto* bo = dyn_cast<BinaryOperator>(clone))
        {
            if ((rng() & 1u) == 0)
            {
                Instruction::BinaryOps newop = bo->getOpcode();
                switch (bo->getOpcode())
                {
                    case Instruction::Add: newop = Instruction::Sub; break;
                    case Instruction::Sub: newop = Instruction::Add; break;
                    case Instruction::And: newop = Instruction::Or; break;
                    case Instruction::Or: newop = Instruction::And; break;
                    case Instruction::Xor: newop = Instruction::Or; break;
                    case Instruction::Mul: newop = Instruction::Add; break;
                    default: break;
                }
                if (newop != bo->getOpcode())
                {
                    IRBuilder<> bob(bo);
                    Value* nb = bob.CreateBinOp(newop, bo->getOperand(0), bo->getOperand(1));
                    bo->replaceAllUsesWith(nb);
                    vmap[&i] = nb;
                    bo->eraseFromParent();
                    ++cloned;
                    continue;
                }
            }
            for (unsigned op = 0; op < bo->getNumOperands(); ++op)
            {
                if (auto* ci = dyn_cast<ConstantInt>(bo->getOperand(op)))
                {
                    if ((rng() % 100u) < 50u)
                    {
                        bo->setOperand(op, ConstantInt::get(ci->getType(), ci->getZExtValue() ^ mutnonce));
                    }
                }
            }
        }

        vmap[&i] = clone;
        ++cloned;
    }

    return cloned;
}

// the empty BB terminator null check must precede isEHPad because BasicBlock::isEHPad null derefs when the BB has no instructions
static void CollectDonors(Function& f, BasicBlock* dispatchhead, const SmallPtrSetImpl<BasicBlock*>& excludes, SmallVectorImpl<BasicBlock*>& out)
{
    for (BasicBlock& bb : f)
    {
        if (&bb == dispatchhead) continue;
        if (!bb.getTerminator()) continue;
        if (bb.isEHPad()) continue;
        if (IsDecoyBlock(&bb)) continue;
        if (excludes.count(&bb)) continue;
        out.push_back(&bb);
    }
}

static void FillDecoyBlock(Function& f, BasicBlock* decoy, BasicBlock* dispatchsrc, BasicBlock* fallthru, ArrayRef<BasicBlock*> donors, DominatorTree& dt, std::mt19937_64& rng, const SmallVectorImpl<PolluteSlot>& slots32, const SmallVectorImpl<PolluteSlot>& slots64, unsigned donorclonecap)
{
    LLVMContext& c = f.getContext();
    IntegerType* i32 = Type::getInt32Ty(c);
    IntegerType* i64 = Type::getInt64Ty(c);

    unsigned diamondcloned = 0;
    if (!donors.empty())
    {
        BasicBlock* donor = donors[rng() % donors.size()];
        diamondcloned = DiamondCloneDonorInto(donor, decoy, dispatchsrc, dt, rng, donorclonecap);
    }

    IRBuilder<> fb(decoy);
    fb.SetInsertPoint(decoy);

    const bool have32 = !slots32.empty();
    const bool have64 = !slots64.empty();

    if (!have32 && !have64)
    {
        (void)EmitSinkLoad(fb, *f.getParent(), i64, "");
        fb.CreateBr(fallthru);
        MarkDecoyBlock(decoy);
        return;
    }

    const unsigned opspan = DecoyMaxTailOps - DecoyMinTailOps + 1u;
    unsigned targetops = DecoyMinTailOps + (unsigned)(rng() % opspan);
    if (diamondcloned >= targetops) targetops = 1;
    else targetops -= diamondcloned;

    Value* acc = nullptr;
    IntegerType* accty = nullptr;

    auto pickslot = [&](IntegerType* ity) -> const PolluteSlot&
    {
        auto& slots = (ity == i64) ? slots64 : slots32;
        return slots[rng() % slots.size()];
    };

    for (unsigned n = 0; n < targetops; ++n)
    {
        IntegerType* ity;
        if (have32 && have64) ity = (rng() & 1) ? i64 : i32;
        else ity = have64 ? i64 : i32;

        if (acc && accty != ity) acc = nullptr;

        if (!acc)
        {
            acc = EmitSlotLoad(fb, pickslot(ity));
            accty = ity;
            continue;
        }

        Value* rhs = EmitSlotLoad(fb, pickslot(ity));
        switch (rng() % 6)
        {
            case 0: acc = fb.CreateXor(acc, rhs, ""); break;
            case 1: acc = fb.CreateAdd(acc, rhs, ""); break;
            case 2: acc = fb.CreateSub(acc, rhs, ""); break;
            case 3: acc = fb.CreateAnd(acc, rhs, ""); break;
            case 4: acc = fb.CreateOr(acc, rhs, ""); break;
            default: acc = fb.CreateMul(acc, rhs, ""); break;
        }

        if ((rng() % 100u) < DecoyStoreChancePct)
        {
            EmitSlotStore(fb, pickslot(ity), acc, false);
        }
    }

    if (acc)
    {
        EmitSlotStore(fb, pickslot(accty), acc, true);
    }

    fb.CreateBr(fallthru);
    MarkDecoyBlock(decoy);
}

// the terminator null check must precede isEHPad because BasicBlock::isEHPad null derefs on empty BBs
static bool IsEHOrFuncletInterior(BasicBlock* bb)
{
    Instruction* term = bb->getTerminator();
    if (!term) return true;
    if (bb->isEHPad()) return true;
    if (isa<CatchReturnInst>(term) || isa<CleanupReturnInst>(term) || isa<ResumeInst>(term)) return true;
    for (Instruction& i : *bb)
    {
        if (auto* cb = dyn_cast<CallBase>(&i))
        {
            if (cb->countOperandBundlesOfType(LLVMContext::OB_funclet) > 0) return true;
        }
    }
    return false;
}

// head sweep starts from the first instruction this pass emitted rather than head->begin() because head's pre existing allocas legitimately host ConstMBA on their user sites and tagging them would suppress legit amplification
static void ApplyBlanketSkipTag(LLVMContext& c, ArrayRef<BasicBlock*> allemittedbbs, Instruction* firstemittedinhead, BasicBlock* headbb)
{
    Metadata* skipops[1] = { MDString::get(c, "retopq_skip") };
    MDNode* skipnode = MDNode::get(c, skipops);

    auto blanket = [&](BasicBlock* target, Instruction* from)
    {
        BasicBlock::iterator it = from ? BasicBlock::iterator(from) : target->begin();
        for (; it != target->end(); ++it)
        {
            MDNode* existing = it->getMetadata("mbagen.escalate");
            if (existing && existing->getNumOperands() > 0)
            {
                if (auto* ms = dyn_cast<MDString>(existing->getOperand(0).get()))
                {
                    if (ms->getString() == "poly_const") continue;
                }
            }
            it->setMetadata("mbagen.escalate", skipnode);
        }
    };

    blanket(headbb, firstemittedinhead);
    for (BasicBlock* b : allemittedbbs)
    {
        if (b == headbb) continue;
        blanket(b, nullptr);
    }
}

struct DispatchPlan
{
    bool HasRealMatch;
    CallInst* RealCall;
};

static BasicBlock* MaterializePostCallBB(CallInst* ci);

// any none call use address-takes the function; invoke's normal dest is not a BlockAddress stable return site because the machine return address goes to the unwind stub; musttail calls are themselves terminators so splitBasicBlock after them is undefined; funclet interior callers break token threading when split
static bool CollectDirectCallersIfSplittable(Function& f, DispatchPlan& out)
{
    out.HasRealMatch = false;
    out.RealCall = nullptr;

    CallInst* thecall = nullptr;
    unsigned callercount = 0;

    for (Use& u : f.uses())
    {
        User* user = u.getUser();
        auto* cb = dyn_cast<CallBase>(user);
        if (!cb) return false;
        if (!cb->isCallee(&u)) return false;
        if (isa<InvokeInst>(cb)) return false;
        auto* ci = dyn_cast<CallInst>(cb);
        if (!ci) return false;
        if (ci->getParent()->getParent() == &f) return false;
        if (IsEHOrFuncletInterior(ci->getParent())) return false;
        if (ci->isMustTailCall()) return false;
        if (!ci->getNextNode()) return false;

        ++callercount;
        if (callercount > 1) return false;
        thecall = ci;
    }

    if (callercount != 1 || !thecall) return false;

    out.HasRealMatch = true;
    out.RealCall = thecall;
    return true;
}

// the machine return address equals the address of the instruction right after the call so splitting at ci->getNextNode() produces a BB whose entry address matches llvm.returnaddress(0) inside the callee
static BasicBlock* MaterializePostCallBB(CallInst* ci)
{
    Instruction* next = ci->getNextNode();
    if (!next) return nullptr;
    BasicBlock* parent = ci->getParent();
    BasicBlock* post = parent->splitBasicBlock(next, "PostCall");
    return post;
}

static bool EmitDispatchChain(Function& f, const DispatchPlan& plan)
{
    if (FunctionAlreadyProcessed(f)) return false;
    if (f.isDeclaration()) return false;
    if (f.empty()) return false;

    BasicBlock& head = f.getEntryBlock();
    if (IsEHOrFuncletInterior(&head)) return false;

    Instruction* lastalloca = nullptr;
    for (Instruction& i : head)
        if (isa<AllocaInst>(&i)) lastalloca = &i;
    Instruction* firstnonalloca = lastalloca ? lastalloca->getNextNode() : head.getFirstNonPHIOrDbgOrLifetime();
    if (!firstnonalloca) return false;
    if (firstnonalloca->isTerminator()) return false;

    LLVMContext& c = f.getContext();
    IntegerType* i32 = Type::getInt32Ty(c);
    IntegerType* i64 = Type::getInt64Ty(c);

    BasicBlock* realbody = head.splitBasicBlock(firstnonalloca, "Real");

    // head has no terminator between splitBr erase and the first conditional branch built below so nothing may call an API that walks head to its end until the new terminator is in place
    Instruction* splitbr = head.getTerminator();
    splitbr->eraseFromParent();

    const unsigned fakespan = FakeCountMax - FakeCountMin + 1u;
    const unsigned chainlen = FakeCountMin + (unsigned)(Rng() % fakespan);
    if (chainlen == 0) return false;

    const int realslot = plan.HasRealMatch ? (int)(Rng() % chainlen) : -1;

    // fakes must stay above 1 to avoid colliding with the common ra of zero case and must dedup against each other so distinct check steps cannot collapse under CSE
    std::vector<uint64_t> fakeconsts;
    fakeconsts.reserve(chainlen);
    std::unordered_set<uint64_t> used;
    for (unsigned i = 0; i < chainlen; ++i)
    {
        if ((int)i == realslot)
        {
            fakeconsts.push_back(0);
            continue;
        }
        uint64_t v;
        do { v = (uint64_t)Rng(); } while (v < 2 || !used.insert(v).second);
        fakeconsts.push_back(v);
    }

    HashConsts hc = RollHashConsts(Rng);

    IRBuilder<> b(&head);
    b.SetInsertPoint(&head);

    Instruction* firstemittedpivot = nullptr;

    Function* rafn = Intrinsic::getDeclaration(f.getParent(), Intrinsic::returnaddress);
    Value* ra = b.CreateCall(rafn, { ConstantInt::get(i32, 0) }, "");
    firstemittedpivot = cast<Instruction>(ra);
    Value* rai64 = b.CreatePtrToInt(ra, i64, "");

    Value* hra = EmitHashIR(b, rai64, i64, hc);

    SmallVector<BasicBlock*, 16> checkbbs;
    checkbbs.reserve(chainlen > 0 ? chainlen - 1 : 0);
    for (unsigned i = 1; i < chainlen; ++i)
    {
        checkbbs.push_back(BasicBlock::Create(c, "Check", &f));
    }

    SmallVector<BasicBlock*, 16> decoybbs;
    decoybbs.reserve(chainlen);
    for (unsigned i = 0; i < chainlen; ++i)
    {
        decoybbs.push_back(BasicBlock::Create(c, "Decoy", &f));
    }

    BasicBlock* postcallbb = nullptr;

    for (unsigned k = 0; k < chainlen; ++k)
    {
        BasicBlock* host = (k == 0) ? &head : checkbbs[k - 1];
        IRBuilder<> ib(host);

        Value* candidate;
        if ((int)k == realslot)
        {
            if (!postcallbb) postcallbb = MaterializePostCallBB(plan.RealCall);
            if (postcallbb)
            {
                BlockAddress* ba = BlockAddress::get(plan.RealCall->getFunction(), postcallbb);
                candidate = ib.CreatePtrToInt(ba, i64, "");
            }
            else
            {
                candidate = ConstantInt::get(i64, (uint64_t)Rng() | 2ull);
            }
        }
        else
        {
            candidate = ConstantInt::get(i64, fakeconsts[k]);
        }

        Value* hcand = EmitHashIR(ib, candidate, i64, hc);

        const uint64_t nonce = Rng();
        const uint64_t esck = Rng();
        Value* noncev = InsertFoldBarrier(ib, ConstantInt::get(i64, nonce));
        Value* lhs = ib.CreateXor(hra, noncev, "");
        lhs = ib.CreateXor(lhs, ConstantInt::get(i64, esck), "");
        Value* rhs = ib.CreateXor(hcand, noncev, "");
        rhs = ib.CreateXor(rhs, ConstantInt::get(i64, esck), "");
        TagEscalateConstMBA(rhs);

        Value* m = ib.CreateICmpEQ(lhs, rhs, "");

        BasicBlock* onmatch = decoybbs[k];
        BasicBlock* onmiss = (k + 1 < chainlen) ? checkbbs[k] : realbody;
        ib.CreateCondBr(m, onmatch, onmiss);
    }

    SmallPtrSet<BasicBlock*, 32> excludes;
    for (BasicBlock* bb : checkbbs) excludes.insert(bb);
    for (BasicBlock* bb : decoybbs) excludes.insert(bb);
    SmallVector<BasicBlock*, 32> donors;
    CollectDonors(f, &head, excludes, donors);

    SmallVector<PolluteSlot, 32> slots32, slots64;
    CollectPolluteSlots(f, i32, i64, slots32, slots64);

    DominatorTree dt;
    dt.recalculate(f);

    for (unsigned k = 0; k < chainlen; ++k)
    {
        BasicBlock* dispatchsrc = (k == 0) ? &head : checkbbs[k - 1];
        FillDecoyBlock(f, decoybbs[k], dispatchsrc, realbody, donors, dt, Rng, slots32, slots64, DonorCloneCap);
    }

    SmallVector<BasicBlock*, 32> allemitted;
    allemitted.push_back(&head);
    for (BasicBlock* bb : checkbbs) allemitted.push_back(bb);
    for (BasicBlock* bb : decoybbs) allemitted.push_back(bb);
    ApplyBlanketSkipTag(c, allemitted, firstemittedpivot, &head);

    MarkFunctionProcessed(f);
    return true;
}

static bool ProcessFunction(Function& f)
{
    if (f.isDeclaration()) return false;
    if (HasNoObfuscateAnnotation(f) || HasNoRetOpaqueAnnotation(f)) return false;
    if (!HasObfuscateAnnotation(f)) return false;
    if (FunctionAlreadyProcessed(f)) return false;

    DispatchPlan plan;
    plan.HasRealMatch = false;
    plan.RealCall = nullptr;
    (void)CollectDirectCallersIfSplittable(f, plan);

    return EmitDispatchChain(f, plan);
}

bool RunRetOpaque(Module& m)
{
    if (!RetOpaqueEnable) return false;

    std::random_device rd;
    uint64_t seed = (uint64_t)rd();
    seed ^= (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count();
    seed ^= ((uint64_t)rd()) << 32;
    seed ^= 0x5245544F50515F5FULL;
    Rng.seed(seed);

    // snapshot the function list before iterating because EmitDispatchChain splits caller BBs which can perturb iteration shape under some module shapes
    SmallVector<Function*, 64> fns;
    for (Function& f : m) fns.push_back(&f);

    bool mutated = false;
    for (Function* f : fns)
    {
        if (ProcessFunction(*f)) mutated = true;
    }
    return mutated;
}

PreservedAnalyses RetOpaquePass::run(Module& m, ModuleAnalysisManager&)
{
    bool mutated = RunRetOpaque(m);
    return mutated ? PreservedAnalyses::none() : PreservedAnalyses::all();
}