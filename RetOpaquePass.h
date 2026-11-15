#pragma once

#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"

struct RetOpaquePass : llvm::PassInfoMixin<RetOpaquePass>
{
    llvm::PreservedAnalyses run(llvm::Module& m, llvm::ModuleAnalysisManager&);
};

bool RunRetOpaque(llvm::Module& m);

extern llvm::cl::opt<bool> RetOpaqueEnable;