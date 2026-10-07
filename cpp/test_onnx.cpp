#include <onnxruntime_cxx_api.h>
#include <iostream>

int main()
{
    Ort::Env env(
        ORT_LOGGING_LEVEL_WARNING,
        "ONNXTest"
    );

    std::cout << "ONNX Runtime C++ is working!" << std::endl;

    return 0;
}