#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>


const char* traffic_sign_names[43] = {
    "Speed limit 20 km/h",
    "Speed limit 30 km/h",
    "Speed limit 50 km/h",
    "Speed limit 60 km/h",
    "Speed limit 70 km/h",
    "Speed limit 80 km/h",
    "End of speed limit 80 km/h",
    "Speed limit 100 km/h",
    "Speed limit 120 km/h",
    "No passing",
    "No passing for vehicles over 3.5 tons",
    "Right-of-way at next intersection",
    "Priority road",
    "Yield",
    "Stop",
    "No vehicles",
    "Vehicles over 3.5 tons prohibited",
    "No entry",
    "General caution",
    "Dangerous curve to the left",
    "Dangerous curve to the right",
    "Double curve",
    "Bumpy road",
    "Slippery road",
    "Road narrows on the right",
    "Road work",
    "Traffic signals",
    "Pedestrians",
    "Children crossing",
    "Bicycles crossing",
    "Beware of ice/snow",
    "Wild animals crossing",
    "End of all speed and passing limits",
    "Turn right ahead",
    "Turn left ahead",
    "Ahead only",
    "Go straight or right",
    "Go straight or left",
    "Keep right",
    "Keep left",
    "Roundabout mandatory",
    "End of no passing",
    "End of no passing for vehicles over 3.5 tons"
};

int main()
{
    // ============================================================
    // 1. File paths
    // ============================================================

    const std::wstring model_path = L"models/traffic_sign_cnn.onnx";

    // Change this later to the path of your traffic sign image
    const std::string image_path = "test_image.jpg";


    // ============================================================
    // 2. Load the image using OpenCV
    // ============================================================

    cv::Mat image = cv::imread(image_path);

    if (image.empty())
    {
        std::cerr << "Error: Could not load image: "
                  << image_path << std::endl;

        return 1;
    }

    std::cout << "Image loaded successfully!" << std::endl;


    // ============================================================
    // 3. Resize image to 32 x 32
    // ============================================================

    cv::Mat resized;

    cv::resize(
        image,
        resized,
        cv::Size(32, 32)
    );


    // ============================================================
    // 4. Convert BGR to RGB
    // ============================================================

    cv::Mat rgb;

    cv::cvtColor(
        resized,
        rgb,
        cv::COLOR_BGR2RGB
    );


    // ============================================================
    // 5. Convert image to float
    // ============================================================

    rgb.convertTo(
        rgb,
        CV_32FC3,
        1.0 / 255.0
    );


    // ============================================================
    // 6. Convert HWC format to CHW format
    //
    // OpenCV image:
    // Height x Width x Channels
    //
    // CNN expects:
    // Channels x Height x Width
    // ============================================================

    std::vector<float> input_tensor_values(
        3 * 32 * 32
    );

    for (int c = 0; c < 3; c++)
    {
        for (int h = 0; h < 32; h++)
        {
            for (int w = 0; w < 32; w++)
            {
                input_tensor_values[
                    c * 32 * 32 +
                    h * 32 +
                    w
                ] = rgb.at<cv::Vec3f>(h, w)[c];
            }
        }
    }


    // ============================================================
    // 7. Create ONNX Runtime environment
    // ============================================================

    Ort::Env env(
        ORT_LOGGING_LEVEL_WARNING,
        "TrafficSignClassifier"
    );


    // ============================================================
    // 8. Create session options
    // ============================================================

    Ort::SessionOptions session_options;

    session_options.SetIntraOpNumThreads(1);

    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED
    );


    // ============================================================
    // 9. Load ONNX model
    // ============================================================

    Ort::Session session(
        env,
        model_path.c_str(),
        session_options
    );

    std::cout << "ONNX model loaded successfully!"
              << std::endl;


    // ============================================================
    // 10. Define input shape
    // ============================================================

    std::vector<int64_t> input_shape = {
        1, 3, 32, 32
    };


    // ============================================================
    // 11. Create memory info
    // ============================================================

    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator,
            OrtMemTypeDefault
        );


    // ============================================================
    // 12. Create input tensor
    // ============================================================

    Ort::Value input_tensor =
        Ort::Value::CreateTensor<float>(
            memory_info,
            input_tensor_values.data(),
            input_tensor_values.size(),
            input_shape.data(),
            input_shape.size()
        );


    // ============================================================
    // 13. Input and output names
    // ============================================================

    const char* input_names[] = {
        "input"
    };

    const char* output_names[] = {
        "output"
    };


    // ============================================================
    // 14. Run inference and measure time
    // ============================================================

    auto start_time =
        std::chrono::high_resolution_clock::now();

    auto output_tensors =
        session.Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input_tensor,
            1,
            output_names,
            1
        );

    auto end_time =
        std::chrono::high_resolution_clock::now();


    // ============================================================
    // 15. Calculate inference time
    // ============================================================

    double inference_time =
        std::chrono::duration<double, std::milli>(
            end_time - start_time
        ).count();


    // ============================================================
    // 16. Get output probabilities
    // ============================================================

    float* output_data =
        output_tensors[0].GetTensorMutableData<float>();


    // ============================================================
    // 17. Find predicted class
    // ============================================================

    int predicted_class = 0;

    float max_value = output_data[0];

    for (int i = 1; i < 43; i++)
    {
        if (output_data[i] > max_value)
        {
            max_value = output_data[i];
            predicted_class = i;
        }
    }


    // ============================================================
    // 18. Convert logits to probabilities using Softmax
    // ============================================================

    float sum_exp = 0.0f;

    for (int i = 0; i < 43; i++)
    {
        sum_exp += std::exp(
            output_data[i] - max_value
        );
    }

    float confidence =
        std::exp(
            output_data[predicted_class] - max_value
        ) / sum_exp;


    // ============================================================
    // 19. Display results
    // ============================================================

    std::cout << std::endl;

    std::cout << "=============================="
              << std::endl;

    std::cout << "Traffic Sign Classification"
              << std::endl;

    std::cout << "=============================="
              << std::endl;

    
    std::cout << "Predicted Class : "
          << predicted_class
          << std::endl;

    std::cout << "Traffic Sign    : "
          << traffic_sign_names[predicted_class]
          << std::endl;

    std::cout << "Confidence      : "
              << confidence * 100
              << "%"
              << std::endl;

    std::cout << "Inference Time  : "
              << inference_time
              << " ms"
              << std::endl;

    std::cout << "=============================="
              << std::endl;


    return 0;
}