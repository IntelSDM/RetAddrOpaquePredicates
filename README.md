# Return Address Opaque Predicates
Built for LLVM 19.1 - Windows x64 <br> <br>

Return Address Opaque Predicates are quite simple. <br>
Functions can return to many different addresses, but those addresses are known at compile time... In an unobfuscated binary that is, once you obfuscate with binary wide obfuscation, removing all the call addresses; its an unknown variable. <br>
This allows us to use them to emit corruption, as the attacker realistically wont know.<br>
There is a caveat, you can technically just dump all call instructions from the binary and bruteforce, but who the fuck would do that on all of the return address checks? <br>
This allows it to defeat AI, SMT, Symbolic Execution and any other static lifting mechanisms. <br>


<img width="1967" height="419" alt="image" src="https://github.com/user-attachments/assets/20787c0c-973b-46af-8f9a-1e298c17260c" /><br>
<img width="1973" height="735" alt="image" src="https://github.com/user-attachments/assets/b2204b21-0f0a-4b6c-92aa-6a8bd440b7a7" /><br>

## How To Build
* Download [LLVM 19.1](https://github.com/llvm/llvm-project/releases/tag/llvmorg-19.1.0)
* Download [VCPKG](https://github.com/microsoft/vcpkg)
* VCPKG install asmjit
* Open build.bat in a text editor.
* Find the LLVM 19.1 references in the command line, Change them to your file path.
* Find the VCPKG references in the command line, Change them to your file path.
* Run the build script
* Build your application with the ``` -load-pass-plugin ``` Param to the FunctionPass.dll
