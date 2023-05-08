### PyTorch demo app

The PyTorch demo app is a full-fledged app that contains two showcases. A camera app that runs a quantized model to classifiy images in real time. And a text-based app that uses a text classification model to predict the topic from the input text.

## PytorchStreamReader failed locating file constants.pkl: file not found

```
2023-04-26 12:42:54.172781+0800 PyTorchDemo[5324:2043559] [SystemGestureGate] <0x139209be0> Gesture: System gesture gate timed out.
2023-04-26 12:42:54.214353+0800 PyTorchDemo[5324:2043952] PytorchStreamReader failed locating file constants.pkl: file not found
Exception raised from valid at /Users/distiller/project/caffe2/serialize/inline_container.cc:157 (most recent call first):
frame #0: _ZN3c106detail14torchCheckFailEPKcS2_jRKNSt3__112basic_stringIcNS3_11char_traitsIcEENS3_9allocatorIcEEEE + 92 (0x103cfbf38 in PyTorchDemo)
frame #1: _ZN6caffe29serialize19PyTorchStreamReader5validEPKcS3_ + 136 (0x1037e7f74 in PyTorchDemo)
frame #2: _ZN6caffe29serialize19PyTorchStreamReader11getRecordIDERKNSt3__112basic_stringIcNS2_11char_traitsIcEENS2_9allocatorIcEEEE + 112 (0x1037e88dc in PyTorchDemo)
frame #3: _ZN6caffe29serialize19PyTorchStreamReader9getRecordERKNSt3__112basic_stringIcNS2_11char_traitsIcEENS2_9allocatorIcEEEE + 76 (0x1037e81ac in PyTorchDemo)
frame #4: _ZN5torch3jit21readArchiveAndTensorsERKNSt3__112basic_stringIcNS1_11char_traitsIcEENS1_9allocatorIcEEEES9_S9_N3c108optionalINS1_8functionIFNSA_13StrongTypePtrERKNSA_13QualifiedNameEEEEEENSB_INSC_IFNSA_13intrusive_ptrINSA_6ivalue6ObjectENSA_6detail34intrusive_target_default_null_typeISM_EEEESD_NSA_6IValueEEEEEENSB_INSA_6DeviceEEERN6caffe29serialize19PyTorchStreamReaderENS1_10shared_ptrINS0_14StorageContextEEE + 196 (0x103ae113c in PyTorchDemo)
frame #5: _ZN5torch3jit12_GLOBAL__N_124ScriptModuleDeserializer11readArchiveERKNSt3__112basic_stringIcNS3_11char_traitsIcEENS3_9allocatorIcEEEE + 308 (0x103ad297c in PyTorchDemo)
frame #6: _ZN5torch3jit12_GLOBAL__N_124ScriptModuleDeserializer11deserializeEN3c108optionalINS3_6DeviceEEERNSt3__113unordered_mapINS7_12basic_stringIcNS7_11char_traitsIcEENS7_9allocatorIcEEEESE_NS7_4hashISE_EENS7_8equal_toISE_EENSC_INS7_4pairIKSE_SE_EEEEEE + 484 (0x103ad09c4 in PyTorchDemo)
frame #7: _ZN5torch3jit4loadENSt3__110shared_ptrIN6caffe29serialize20ReadAdapterInterfaceEEEN3c108optionalINS7_6DeviceEEERNS1_13unordered_mapINS1_12basic_stringIcNS1_11char_traitsIcEENS1_9allocatorIcEEEESH_NS1_4hashISH_EENS1_8equal_toISH_EENSF_INS1_4pairIKSH_SH_EEEEEE + 508 (0x103ad2250 in PyTorchDemo)
frame #8: _ZN5torch3jit4loadERKNSt3__112basic_stringIcNS1_11char_traitsIcEENS1_9allocatorIcEEEEN3c108optionalINSA_6DeviceEEERNS1_13unordered_mapIS7_S7_NS1_4hashIS7_EENS1_8equal_toIS7_EENS5_INS1_4pairIS8_S7_EEEEEE + 112 (0x103ad254c in PyTorchDemo)
frame #9: _ZN5torch3jit4loadERKNSt3__112basic_stringIcNS1_11char_traitsIcEENS1_9allocatorIcEEEEN3c108optionalINSA_6DeviceEEE + 44 (0x103ad24ac in PyTorchDemo)
frame #10: -[TorchModule + + (0x103c97678 in PyTorchDemo)
frame #11: $sSo17VisionTorchModuleC10fileAtPathABSgSS_tcfcTO + 48 (0x103cae5d8 in PyTorchDemo)
frame #12: $sSo17VisionTorchModuleC10fileAtPathABSgSS_tcfC + 52 (0x103cad580 in PyTorchDemo)
frame #13: $s11PyTorchDemo14MUSIQPredictorC6module33_B73206B42ECC991607014723026A6B15LLSo06VisionB6ModuleCvgAGyXEfU_ + 484 (0x103cad488 in PyTorchDemo)
frame #14: $s11PyTorchDemo14MUSIQPredictorC6module33_B73206B42ECC991607014723026A6B15LLSo06VisionB6ModuleCvg + 156 (0x103cad230 in PyTorchDemo)
frame #15: $s11PyTorchDemo14MUSIQPredictorC7predict_11resultCountSayAA15InferenceResultVG_SdtSgSaySfG_SitKF + 308 (0x103cadd58 in PyTorchDemo)
frame #16: $s11PyTorchDemo19MUSIQViewControllerC15runMUSIQPredictyySo7UIImageCFyycfU0_ + 340 (0x103cc8fa0 in PyTorchDemo)
frame #17: $sIeg_IeyB_TR + 48 (0x103c9f904 in PyTorchDemo)
frame #18: _dispatch_call_block_and_release + 32 (0x105aac598 in libdispatch.dylib)
frame #19: _dispatch_client_callout + 20 (0x105aae04c in libdispatch.dylib)
frame #20: _dispatch_queue_override_invoke + 1052 (0x105ab0b84 in libdispatch.dylib)
frame #21: _dispatch_root_queue_drain + 408 (0x105ac2468 in libdispatch.dylib)
frame #22: _dispatch_worker_thread2 + 196 (0x105ac2e64 in libdispatch.dylib)
frame #23: _pthread_wqthread + 228 (0x224664dbc in libsystem_pthread.dylib)
frame #24: start_wqthread + 8 (0x224664b98 in libsystem_pthread.dylib)
```


## Is torch::jit::load (LibTorch) supposed to work with torch.save (PyTorch)?

https://github.com/pytorch/pytorch/issues/47917

torch::load("model.pt") should indeed be torch::jit::load("model.pt") (I missed the jit.). The stack trace contains references to torch::jit::load and torch::load does not return a value.

Is torch::jit::load (LibTorch) supposed to work with torch.save (PyTorch)?

No, this is not meant to work and making that work is not the intention of converging the serialization container formats. We should probably re-word the documentation to make this more clear.

## LOADING A TORCHSCRIPT MODEL IN C++

https://pytorch.org/tutorials/advanced/cpp_export.html

> Converting Your PyTorch Model to Torch Script

## Convert Image to CVPixelBuffer for Machine Learning Swift

https://stackoverflow.com/questions/44400741/convert-image-to-cvpixelbuffer-for-machine-learning-swift

```
    DispatchQueue.global(qos: .userInitiated).async {
        // Resnet50 expects an image 224 x 224, so we should resize and crop the source image
        let inputImageSize: CGFloat = 224.0
        let minLen = min(image.size.width, image.size.height)
        let resizedImage = image.resize(to: CGSize(width: inputImageSize * image.size.width / minLen, height: inputImageSize * image.size.height / minLen))
        let cropedToSquareImage = resizedImage.cropToSquare()

        guard let pixelBuffer = cropedToSquareImage?.pixelBuffer() else {
            fatalError()
        }
        guard let classifierOutput = try? self.classifier.prediction(image: pixelBuffer) else {
            fatalError()
        }

        DispatchQueue.main.async {
            self.title = classifierOutput.classLabel
        }
    }

// ...

extension UIImage {

    func resize(to newSize: CGSize) -> UIImage {
        UIGraphicsBeginImageContextWithOptions(CGSize(width: newSize.width, height: newSize.height), true, 1.0)
        self.draw(in: CGRect(x: 0, y: 0, width: newSize.width, height: newSize.height))
        let resizedImage = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()

        return resizedImage
    }

    func cropToSquare() -> UIImage? {
        guard let cgImage = self.cgImage else {
            return nil
        }
        var imageHeight = self.size.height
        var imageWidth = self.size.width

        if imageHeight > imageWidth {
            imageHeight = imageWidth
        }
        else {
            imageWidth = imageHeight
        }

        let size = CGSize(width: imageWidth, height: imageHeight)

        let x = ((CGFloat(cgImage.width) - size.width) / 2).rounded()
        let y = ((CGFloat(cgImage.height) - size.height) / 2).rounded()

        let cropRect = CGRect(x: x, y: y, width: size.height, height: size.width)
        if let croppedCgImage = cgImage.cropping(to: cropRect) {
            return UIImage(cgImage: croppedCgImage, scale: 0, orientation: self.imageOrientation)
        }

        return nil
    }

    func pixelBuffer() -> CVPixelBuffer? {
        let width = self.size.width
        let height = self.size.height
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
                     kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                         Int(width),
                                         Int(height),
                                         kCVPixelFormatType_32ARGB,
                                         attrs,
                                         &pixelBuffer)

        guard let resultPixelBuffer = pixelBuffer, status == kCVReturnSuccess else {
            return nil
        }

        CVPixelBufferLockBaseAddress(resultPixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(resultPixelBuffer)

        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(data: pixelData,
                                      width: Int(width),
                                      height: Int(height),
                                      bitsPerComponent: 8,
                                      bytesPerRow: CVPixelBufferGetBytesPerRow(resultPixelBuffer),
                                      space: rgbColorSpace,
                                      bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue) else {
                                        return nil
        }

        context.translateBy(x: 0, y: height)
        context.scaleBy(x: 1.0, y: -1.0)

        UIGraphicsPushContext(context)
        self.draw(in: CGRect(x: 0, y: 0, width: width, height: height))
        UIGraphicsPopContext()
        CVPixelBufferUnlockBaseAddress(resultPixelBuffer, CVPixelBufferLockFlags(rawValue: 0))

        return resultPixelBuffer
    }
}
```

```
import Vision
import CoreML

let model = try VNCoreMLModel(for: MyCoreMLGeneratedModelClass().model)
let request = VNCoreMLRequest(model: model, completionHandler: myResultsMethod)
let handler = VNImageRequestHandler(url: myImageURL)
handler.perform([request])

func myResultsMethod(request: VNRequest, error: Error?) {
    guard let results = request.results as? [VNClassificationObservation]
        else { fatalError("huh") }
    for classification in results {
        print(classification.identifier, // the scene label
              classification.confidence)
    }

}
```

## a Tensor with 100352 elements cannot be converted to Scalar

```
- (NSArray<NSNumber*>*)predictImage:(void*)imageBuffer {
    try {
        at::Tensor tensor = torch::from_blob(imageBuffer, {1, 3, 224, 224}, at::kFloat);
        torch::autograd::AutoGradMode guard(false);
        at::AutoNonVariableTypeMode non_var_type_mode(true);
        auto outputTensor = _impl.forward({tensor}).toTensor();
        at::Scalar item = outputTensor.item();//test
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
```

```
a Tensor with 100352 elements cannot be converted to Scalar

100352 = 224 * 224 * 2

at::Scalar item = outputTensor.item();
```

## Lite Interpreter version number does not match. The model version must be between 3 and 7 but the model version is 8 ()

https://stackoverflow.com/questions/71379743/any-idea-of-to-solve-the-version-problem-of-pytorch-model-on-android-device-the

```
convert2version5 = True
if convert2version5:
    from torch.jit.mobile import (
        _backport_for_mobile,
        _get_model_bytecode_version,
    )

    MODEL_INPUT_FILE = "model_v7.ptl"
    MODEL_OUTPUT_FILE = "model_v5.ptl"

    print("model version", _get_model_bytecode_version(f_input=MODEL_INPUT_FILE))

    _backport_for_mobile(f_input=MODEL_INPUT_FILE, f_output=MODEL_OUTPUT_FILE, to_version=5)

    print("new model version", _get_model_bytecode_version(MODEL_OUTPUT_FILE))
```

## inline at::Tensor from_blob

```
/// Exposes the given `data` as a `Tensor` without taking ownership of the
/// original data. `sizes` should specify the shape of the tensor. The
/// `TensorOptions` specify additional configuration options for the returned
/// tensor, such as what type to interpret the `data` as.
inline at::Tensor from_blob(
    void* data,
    at::IntArrayRef sizes,
    const at::TensorOptions& options = at::TensorOptions()) {
  at::Tensor tensor = ([&]() {
    at::AutoDispatchBelowAutograd guard;  // TODO: remove
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    return at::from_blob(data, sizes, options.requires_grad(c10::nullopt));
  })();
  return autograd::make_variable(tensor, options.requires_grad());
}
```

https://www.jianshu.com/p/7cddc09ca7a4?tdsourcetag=s_pctim_aiomsg


https://github.com/pytorch/ios-demo-app/blob/master/SpeechRecognition/SpeechRecognition/InferenceModule.mm

## toStringRef
```
auto result = _impl.forward({ tensor }).toStringRef();
NSString *score = [NSString stringWithCString:result.c_str() encoding:[NSString defaultCStringEncoding]];
```

## torch::ones

```
torch::ones(<#at::IntArrayRef size#>, <#c10::optional<at::DimnameList> names#>)
```

## pod search LibTorch

```
-> LibTorch (1.13.0.1)
   The PyTorch C++ library for iOS
   pod 'LibTorch', '~> 1.13.0.1'
   - Homepage: https://github.com/pytorch/pytorch
   - Source:   https://ossci-ios.s3.amazonaws.com/libtorch_ios_1.13.0.zip
   - Versions: 1.13.0.1, 1.13.0, 1.12.0, 1.11.0, 1.10.0, 1.9.0, 1.8.0, 1.7.1, 1.7.0, 1.6.1, 1.6.0, 1.5.0, 1.4.0, 1.3.1, 1.3.0, 0.0.2, 0.0.1 [master
   repo]
   - Subspecs:
     - LibTorch/Core (1.13.0.1)
     - LibTorch/Torch (1.13.0.1)

-> LibTorch-Lite (1.13.0.1)
   The PyTorch C++ library for iOS
   pod 'LibTorch-Lite', '~> 1.13.0.1'
   - Homepage: https://github.com/pytorch/pytorch
   - Source:   https://ossci-ios.s3.amazonaws.com/libtorch_lite_ios_1.13.0.zip
   - Versions: 1.13.0.1, 1.13.0, 1.12.0, 1.11.0, 1.10.0, 1.9.0 [master repo]
   - Subspecs:
     - LibTorch-Lite/Core (1.13.0.1)
     - LibTorch-Lite/Torch (1.13.0.1)

-> LibTorch-Lite-Dummy (1.9.1)
   The PyTorch C++ library for iOS
   pod 'LibTorch-Lite-Dummy', '~> 1.9.1'
   - Homepage: https://github.com/pytorch/pytorch
   - Source:   https://ossci-ios.s3.amazonaws.com/libtorch_lite_ios_1.9.0.zip
   - Versions: 1.9.1, 1.9.0 [master repo]
   - Subspecs:
     - LibTorch-Lite-Dummy/Core (1.9.1)
     - LibTorch-Lite-Dummy/Torch (1.9.1)

-> LibTorch-Lite-Nightly (1.14.0.20221109)
   The nightly build version of PyTorch C++ library for iOS
   pod 'LibTorch-Lite-Nightly', '~> 1.14.0.20221109'
   - Homepage: https://github.com/pytorch/pytorch
   - Source:   https://ossci-ios-build.s3.amazonaws.com/libtorch_lite_ios_nightly_1.14.0.20221109.zip
   - Versions: 1.14.0.20221109, 1.14.0.20221108, 1.14.0.20221107, 1.14.0.20221106, 1.14.0.20221104, 1.14.0.20221102, 1.14.0.20221101, 1.14.0.20221031,
```
