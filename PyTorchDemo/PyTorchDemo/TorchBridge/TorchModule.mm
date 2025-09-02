#import "TorchModule.h"
#import <LibTorch/LibTorch.h>

@implementation TorchModule {
@protected
    torch::jit::script::Module _impl;
    torch::jit::script::Module _sub_impl;
}

- (nullable instancetype)initWithFileAtPath:(NSString*)filePath {
    self = [super init];
    if (self) {
        try {
            auto qengines = at::globalContext().supportedQEngines();
            if (std::find(qengines.begin(), qengines.end(), at::QEngine::QNNPACK) != qengines.end()) {
                at::globalContext().setQEngine(at::QEngine::QNNPACK);
            }
            _impl = torch::jit::load(filePath.UTF8String);
            
            // The state dictionary should now be loaded into the model
            // You can verify this by printing the model's parameters
//            for (const auto& param : _impl.parameters(true)) {
//                std::cout << "_impl.parameters:" << std::endl;
//                std::cout << param << std::endl;
//            }
            
            //            _impl = torch::load(filePath.UTF8String);
            _impl.eval();
        } catch (const std::exception& exception) {
            NSLog(@"%s", exception.what());
            return nil;
        }
    }
    return self;
}

- (nullable instancetype)initWithFileAtPath:(NSString*)filePath subFilePath:(NSString*)subFilePath {
    self = [super init];
    if (self) {
        try {
            auto qengines = at::globalContext().supportedQEngines();
            if (std::find(qengines.begin(), qengines.end(), at::QEngine::QNNPACK) != qengines.end()) {
                at::globalContext().setQEngine(at::QEngine::QNNPACK);
            }
            _impl = torch::jit::load(filePath.UTF8String);
            _sub_impl = torch::jit::load(subFilePath.UTF8String);
            
            // The state dictionary should now be loaded into the model
            // You can verify this by printing the model's parameters
//            for (const auto& param : _impl.parameters(true)) {
//                std::cout << "_impl.parameters:" << std::endl;
//                std::cout << param << std::endl;
//            }
            
            //            _impl = torch::load(filePath.UTF8String);
            _impl.eval();
            _sub_impl.eval();
        } catch (const std::exception& exception) {
            NSLog(@"%s", exception.what());
            return nil;
        }
    }
    return self;
}

@end

@implementation MUSIQTorchModule

- (NSDictionary *)configuration {
    return @{@"n_enc_seq": @(32*24 + 12*9 + 7*5), //input feature map dimension (N = H*W) from backbone // 911
             @"batch_size" : @(1), // fix the value as 1 (for inference)
    };
}

- (float)predictImage:(void*)imageBuffer
                 size:(CGSize)size
           scaledImg1:(void*)scaledImageBuffer1
                size1:(CGSize)size1
           scaledImg2:(void*)scaledImageBuffer2
                size2:(CGSize)size2 {
    try {
        //        double array[] = { 1, 2, 3, 4, 5};
        //        auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA, 1);
        //        auto options = torch::TensorOptions().dtype(torch::kFloat);
        //        auto options = torch::TensorOptions().requires_grad(true);
        //        torch::Tensor tharray = torch::from_blob(array, {5}, options);
        at::Tensor tensor = torch::from_blob(imageBuffer, {1, 3, static_cast<long long>(size.height), static_cast<long long>(size.width)}, at::kFloat);
        //        at::Tensor tensor = torch::from_blob(imageBuffer, {1, 3, 224, 224}, options);
        at::Tensor scaledTensor1 = torch::from_blob(scaledImageBuffer1, {1, 3, static_cast<long long>(size1.height), static_cast<long long>(size1.width)}, at::kFloat);
        at::Tensor scaledTensor2 = torch::from_blob(scaledImageBuffer2, {1, 3, static_cast<long long>(size2.height), static_cast<long long>(size2.width)}, at::kFloat);
        
        //test
        //let maskInputs = torch.ones([config.batchSize, config.nEncSeq+1], device: config.device)
        
        //test
        // convert image data to Tensor
        // tensor = tensor.permute({2, 0, 1});
        // tensor = tensor.toType(torch::kFloat);
        // tensor = tensor.div(255);
        // tensor = tensor.unsqueeze(0); //# add a new dimension at position 0
        
        NSDictionary *config = [self configuration];
        int batchSize = [[config valueForKey:@"batch_size"] intValue];
        int n_enc_seq = [[config valueForKey:@"n_enc_seq"] intValue];
        auto mask_inputs = at::ones({batchSize, n_enc_seq+1}); // 911+1=912
        //        torch::ones(at::IntArrayRef size)
        //        torch::ones(@[@(batchSize), @(n_enc_seq + 1)]);
        //        var maskInputs = torch.ones([config.batchSize, config.nEncSeq+1], device: config.device);
        
        torch::autograd::AutoGradMode guard(false);
        at::AutoNonVariableTypeMode non_var_type_mode(true);
        
        //test
        //        auto result = _impl.forward({ tensor }).toStringRef();
        //        NSString *score = [NSString stringWithCString:result.c_str() encoding:[NSString defaultCStringEncoding]];
        
        //        Expected at most 2 argument(s) for operator 'forward', but received 5 argument(s). Declaration: forward(__torch__.model.backbone.___torch_mangle_740.ResNetBackbone self, Tensor x) -> (Tensor)
        //        auto result = _impl.forward({ mask_inputs, tensor, scaledTensor1, scaledTensor2 }).toStringRef();
        
        // Concatenate the tensors along the first dimension.
        //        torch.cat(): Tensors must have same number of dimensions: got 2 and 4
        //        torch::Tensor merged_tensor = torch::cat({mask_inputs, tensor, scaledTensor1, scaledTensor2}, /*dim=*/0);
        //        auto result = _impl.forward({ merged_tensor }).toStringRef();
        
        // Call the forward method of the C++ script module to make predictions
        /*
         Expected at most 2 argument(s) for operator 'forward', but received 5 argument(s). Declaration: forward(__torch__.model.backbone.___torch_mangle_740.ResNetBackbone self, Tensor x) -> (Tensor)
         */
        
        //test
        // Preprocess the image tensor
        // Convert from RGBA to RGB and normalize
//        tensor = tensor.permute({0, 3, 1, 2}).to(torch::kFloat);
//        tensor = tensor.slice(1, 0, 3);
//        tensor = tensor.div(255).sub(0.5).div(0.5);
        
//        auto outputTensor = _impl.forward({tensor}).toTensor();
        
        //test
        /*
         https://stackoverflow.com/questions/63502473/different-output-from-libtorch-c-and-pytorch
         before the final normalization, you need to scale your input to the range 0-1 and then carry on the normalization you are doing. convert to float and then divide by 255 should get you there.
         */
        // number of dims don't match in permute
//        tensor = tensor.to(at::kFloat).div(255).unsqueeze(0);
//        tensor = tensor.permute({ 0, 3, 1, 2 });
//        tensor.sub_(0.5).div_(0.5);
        
        /*
         OpenCV img = cv2.imread(path) loads an image with HWC-layout (height, width, channels), while Pytorch requires CHW-layout. So we have to do np.transpose(image,(2,0,1)) for HWC->CHW transformation.
         */
//        tensor = tensor.permute({2, 0, 1});
//        scaledTensor1 = scaledTensor1.permute({2, 0, 1});
//        scaledTensor2 = scaledTensor2.permute({2, 0, 1});
        
        tensor = tensor.to(torch::kCPU);
        scaledTensor1 = scaledTensor1.to(torch::kCPU);
        scaledTensor2 = scaledTensor2.to(torch::kCPU);
        
        // Evaluate the input tensor
        at::Tensor outputTensor = _impl.forward({tensor}).toTensor();
        at::Tensor outputTensor1 = _impl.forward({scaledTensor1}).toTensor();
        at::Tensor outputTensor2 = _impl.forward({scaledTensor2}).toTensor();
        
        // Reduce the dimensions of the output tensor
        /*
         view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
         */
//        outputTensor = outputTensor.view({-1});
        
        // Reshape the output tensor into a 1D vector
//        outputTensor = outputTensor.reshape({-1});
        
        // Get the predicted IQA score
        /*
         a Tensor with 65536 elements cannot be converted to Scalar
         */
//        float iqa_score = outputTensor.item<float>();
        
        // Perform average pooling to reduce the feature map to a scalar per sample
//        outputTensor = torch::mean(outputTensor, {2, 3});

        // Convert the output tensor to a vector
        /*
         TensorAccessor expected 1 dims but tensor has 2
         */
//        std::vector<float> iqa_scores(outputTensor.size(0));
//        float* output_data = outputTensor.accessor<float, 1>().data();
//        for (int i = 0; i < outputTensor.size(0); ++i) {
//          iqa_scores[i] = output_data[i];
//        }

        // For a single sample, get the predicted IQA score
//        float iqa_score = iqa_scores[0];
        
//        std::cout << "outputTensor before flatten to 1D vector:\n" << outputTensor << std::endl;
    
//        outputTensor = outputTensor.cpu();//test
//        std::cout << "Embds: " << outputTensor << std::endl;
        
        // Flatten the output tensor to a 1D vector
//        outputTensor = outputTensor.flatten();

        // Get the predicted IQA score as a scalar value
//        float iqa_score = outputTensor[0].item<float>();
//        std::cout << "iqa_score:" << iqa_score << std::endl;
        
        /*
         Expected at most 2 argument(s) for operator 'forward', but received 5 argument(s). Declaration: forward(__torch__.model.backbone.___torch_mangle_740.ResNetBackbone self, Tensor x) -> (Tensor)
         */
//        auto outputTensor = _impl.forward({inputs}).toTensor();
        
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(mask_inputs);
        inputs.push_back(outputTensor);
        inputs.push_back(outputTensor1);
        inputs.push_back(outputTensor2);
        
        // RuntimeError: The size of tensor a (24) must match the size of tensor b (32) at non-singleton dimension 3
        // "n_enc_seq": @(32*24 + 12*9 + 7*5), //input feature map dimension (N = H*W) from backbone
        /*
         feat_dis_org.shape: torch.Size([1, 2048, 24, 32])
         feat_dis_scale_1.shape: torch.Size([1, 2048, 9, 12])
         feat_dis_scale_2.shape: torch.Size([1, 2048, 5, 7])
         */
        /*
         feat_dis_org:[1, 2048, 32, 24]
         feat_dis_scale_1:[1, 2048, 12, 9]
         feat_dis_scale_2:[1, 2048, 7, 6]
         */
        /*
         32/24: 1024 * 768 = 786432, 786432 * 2 = 1572864
         12/9: 384 * 288 = 110592, 110592 * 2 = 221184
         7/5: 224 * 160 = 35840, 35840 * 2 = 71680
         */
        std::cout << "feat_dis_org:" << outputTensor.sizes() << std::endl;
        std::cout << "feat_dis_scale_1:" << outputTensor1.sizes() << std::endl;
        std::cout << "feat_dis_scale_2:" << outputTensor2.sizes() << std::endl;
        
        auto iqaOutput = _sub_impl.forward({inputs}).toTensor();
        float iqa_score = iqaOutput.item<float>();
        std::cout << "iqa_score:" << iqa_score << std::endl;
        //Scores range from 1 to 5, same as scores from Koniq-10k dataset
        //converted to 0~100
        float iqa_score_range_reset = (iqa_score - 1) / 4.0 * 100;
        return iqa_score_range_reset;
        
        //        torch::jit::Object::Property item = _impl.get_property("item");
        //        auto genericList = _impl.run_method("item");
        // Exception: a Tensor with 100352 elements cannot be converted to Scalar
        //        auto item = outputTensor.item<float>();
        
        // Concatenate the pred tensor with the pred_total tensor
        // Initialize the pred_total tensor
        //        torch::Tensor pred_total = torch::zeros({0});
        //        int dim = 4; // specify the appropriate dimension for concatenation
        //        pred_total = torch::cat({pred_total, outputTensor}, dim);
        
        // Convert the pred tensor to a float tensor and append it to the pred_total tensor
        /*
         Could not run 'aten::cat' with arguments from the 'NestedTensor' backend. This could be because the operator doesn't exist for this backend, or was omitted during the selective/custom build process (if using custom build). If you are a Facebook employee using PyTorch on mobile, please visit https://fburl.com/ptmfixes for possible resolutions. 'aten::cat' is only available for these backends: [Undefined, CPU, CUDA, HIP, FPGA, MSNPU, XLA, MLC, Vulkan, Metal, XPU, HPU, Meta, QuantizedCPU, QuantizedCUDA, CustomRNGKeyId, MkldnnCPU, SparseCPU, SparseCUDA, SparseHIP, SparseCsrCPU, SparseCsrCUDA, PrivateUse1, PrivateUse2, PrivateUse3, BackendSelect, Named, ADInplaceOrView, AutogradOther, AutogradCPU, AutogradCUDA, AutogradXLA, UNKNOWN_TENSOR_TYPE_ID, AutogradMLC, Tracer, Autocast, Batched, VmapMode].
         */
        //        pred_total = torch::cat({pred_total, outputTensor.toType(torch::kFloat)}, dim);
        
//        auto floatOutputTensor = outputTensor.toType(torch::kFloat);
        
        // Print the output tensor
        /*
         [ CPUFloatType{1,2048,7,6} ]
         */
//        std::cout << "outputTensor:\n" << outputTensor.sizes() << std::endl;
//        std::cout << "floatOutputTensor:\n" << floatOutputTensor << std::endl;
        
        //test
        /*
         1024 * 768 = 786432, 786432 * 2 = 1572864
         384 * 288 = 110592, 110592 * 2 = 221184
         224 * 168 = 37632, 37632 * 2 = 75264
         
         a Tensor with 100352 elements cannot be converted to Scalar
         a Tensor with 1572864 elements cannot be converted to Scalar
         */
        //        auto item = floatOutputTensor.item<float>();
        
        //        auto max_result = outputTensor.max(1,true);
        //        auto max_index = std::get<1>(max_result).item<float>();
        
//        float* floatBuffer = outputTensor.data_ptr<float>();
//        if (!floatBuffer) {
//            return 0;
//        }
//        NSMutableArray* results = [[NSMutableArray alloc] init];
//        CGFloat sum = 0;//802716.51976261614
//        for (int i = 0; i < size2.width*size2.height*2; i++) { // inputImageWidth2
//            [results addObject:@(floatBuffer[i])];
//            sum += floatBuffer[i];
//        }
//        return [results copy];
    } catch (const std::exception& exception) {
        NSLog(@"%s", exception.what());
    }
    return 0;
}

@end

@implementation VisionTorchModule

- (NSArray<NSNumber*>*)predictImage:(void*)imageBuffer {
    try {
        at::Tensor tensor = torch::from_blob(imageBuffer, {1, 3, 224, 224}, at::kFloat);
        torch::autograd::AutoGradMode guard(false);
        at::AutoNonVariableTypeMode non_var_type_mode(true);
        auto outputTensor = _impl.forward({tensor}).toTensor();
        float* floatBuffer = outputTensor.data_ptr<float>();
        if (!floatBuffer) {
            return nil;
        }
        NSMutableArray* results = [[NSMutableArray alloc] init];
        for (int i = 0; i < 1000; i++) {
            [results addObject:@(floatBuffer[i])];
        }
        return [results copy];
    } catch (const std::exception& exception) {
        NSLog(@"%s", exception.what());
    }
    return nil;
}

@end

@implementation TeacherTorchModule

- (float)predictImage:(void*)imageBuffer
                 size:(CGSize)size {
    try {
        // Create input tensor from image buffer with pinned memory
        auto options = torch::TensorOptions()
            .dtype(at::kFloat)
            .pinned_memory(true);
        
        at::Tensor tensor = torch::from_blob(imageBuffer, 
            {1, 3, static_cast<long long>(size.height), static_cast<long long>(size.width)}, 
            options);
            
        // Move to CPU and ensure contiguous memory layout
        tensor = tensor.to(torch::kCPU).contiguous();
        
        // Use RAII for tensor memory management
        {
            // Disable gradient computation for inference
            torch::autograd::AutoGradMode guard(false);
            at::AutoNonVariableTypeMode non_var_type_mode(true);
            
            // Run model inference with memory optimization
            auto outputTensor = _impl.forward({tensor}).toTensor();
            
            // Get the predicted score and immediately release tensor memory
            float score = outputTensor.item<float>();
            
            // Explicitly clear tensors
            tensor.reset();
            outputTensor.reset();
            
            // Convert to 0-100 range for display
            return score * 100.0f;
        }
        
    } catch (const std::exception& exception) {
        NSLog(@"%s", exception.what());
    }
    return 0;
}

@end

@implementation NLPTorchModule

- (NSArray<NSNumber*>*)predictText:(NSString*)text {
    try {
        const char* buffer = text.UTF8String;
        torch::autograd::AutoGradMode guard(false);
        at::AutoNonVariableTypeMode non_var_type_mode(true);
        at::Tensor tensor = torch::from_blob((void*)buffer, {1, (int64_t)(strlen(buffer))}, at::kByte);
        auto outputTensor = _impl.forward({tensor}).toTensor();
        float* floatBuffer = outputTensor.data_ptr<float>();
        if (!floatBuffer) {
            return nil;
        }
        NSMutableArray* results = [[NSMutableArray alloc] init];
        for (int i = 0; i < 16; i++) {
            [results addObject:@(floatBuffer[i])];
        }
        return [results copy];
    } catch (const std::exception& exception) {
        NSLog(@"%s", exception.what());
    }
    return nil;
}

- (NSArray<NSString*>*)topics {
    try {
        auto genericList = _impl.run_method("get_classes").toList();
        NSMutableArray<NSString*>* topics = [NSMutableArray<NSString*> new];
        for (int i = 0; i < genericList.size(); i++) {
            std::string topic = genericList.get(i).toString()->string();
            [topics addObject:[NSString stringWithCString:topic.c_str() encoding:NSUTF8StringEncoding]];
        }
        return [topics copy];
    } catch (const std::exception& exception) {
        NSLog(@"%s", exception.what());
    }
    return nil;
}

@end
