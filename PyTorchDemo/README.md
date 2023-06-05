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

## Core ML Sample

https://github.com/yulingtianxia/Core-ML-Sample

## swift load coreml of resnet50

**rootModelURL**:

file:///private/var/containers/Bundle/Application/5F1B30F7-663D-4364-8C0D-139829F36E22/ScoreImage.app/resnet50_ML_Neural_Network.mlmodelc/
 
```
func loadResNet50Model() -> VNCoreMLModel? {
    guard let modelURL = Bundle.main.url(forResource: "Resnet50", withExtension: "mlmodelc") else {
        print("Failed to find Resnet50 model in bundle.")
        return nil
    }

    do {
        let coreMLModel = try MLModel(contentsOf: modelURL)
        let visionModel = try VNCoreMLModel(for: coreMLModel)
        return visionModel
    } catch {
        print("Error loading Resnet50 model: \(error)")
        return nil
    }
}
```

```
2023-06-01 14:51:41.171766+0800 ScoreImage[714:90184] Metal API Validation Enabled
Error loading Resnet50 model: Error Domain=com.apple.vis Code=15 "The model does not have a valid input feature of type image" UserInfo={NSLocalizedDescription=The model does not have a valid input feature of type image}
```

```
let resnet50MLModel = resnet50_ML_Neural_Network()
        
(lldb) po resnet50MLModel
<resnet50_ML_Neural_Network: 0x281137ba0>

(lldb) po resnet50MLModel.model

modelDescription: 
inputs: (
    "x : MultiArray (Float32, 1 \U00d7 3 \U00d7 224 \U00d7 224)"
)
outputs: (
    "var_830 : MultiArray (Float32, )"
)
predictedFeatureName: (null)
predictedProbabilitiesName: (null)
updatable: NO
trainingInputs: (
)
parameters: (
)
metadata: {
    MLModelAuthorKey = "PlanetArt: GavinXiang";
    MLModelCreatorDefinedKey =     {
        "com.github.apple.coremltools.source" = "torch==2.1.0.dev20230422";
        "com.github.apple.coremltools.version" = "6.3.0";
    };
    MLModelDescriptionKey = "ResNet-50 from Deep Residual Learning for Image Recognition (paper link: https://arxiv.org/abs/1512.03385).";
    MLModelLicenseKey = "MIT License.";
    MLModelVersionStringKey = "1.0.0";
}, 
configuration: 
 computeUnits: All,             
 useWatchSPIForScribble: NO,             
 allowLowPrecisionAccumulationOnGPU: NO,             
 allowBackgroundGPUComputeSetting: NO,             
 preferredMetalDevice: (null),             
 enableTestVectorMode: NO,             
 parameters: (null),             
 rootModelURL: file:///private/var/containers/Bundle/Application/5F1B30F7-663D-4364-8C0D-139829F36E22/ScoreImage.app/resnet50_ML_Neural_Network.mlmodelc/,             
 profilingOptions: 0,             
 usePreloadedKey: NO,             
 trainWithMLCompute: NO,             
 parentModelName: ,             
 modelName: resnet50_ML_Neural_Network,             
```

## Very slow to load core ML of IQA

```
2023-06-02 16:31:03.531983+0800 ScoreImage[22728:1556762] [default] LSPrefs: could not find untranslocated node for <FSNode 0x600001ec80c0> { isDir = ?, path = '/private/var/folders/wk/frkkcch539lc6s2dk6dw9dy80000gn/X/F7776212-4447-5E1D-BADF-D23F26ACA647/d/Wrapper/ScoreImage.app' }, proceeding on the assumption it is not translocated: Error Domain=NSPOSIXErrorDomain Code=1 "Operation not permitted"
2023-06-02 16:31:03.535646+0800 ScoreImage[22728:1556762] [default] LSPrefs: could not find untranslocated node for <FSNode 0x600001ec80c0> { isDir = ?, path = '/private/var/folders/wk/frkkcch539lc6s2dk6dw9dy80000gn/X/F7776212-4447-5E1D-BADF-D23F26ACA647/d/Wrapper/ScoreImage.app' }, proceeding on the assumption it is not translocated: Error Domain=NSPOSIXErrorDomain Code=1 "Operation not permitted"
2023-06-02 16:31:03.539033+0800 ScoreImage[22728:1556762] [default] LSPrefs: could not find untranslocated node for <FSNode 0x600001ec80c0> { isDir = ?, path = '/private/var/folders/wk/frkkcch539lc6s2dk6dw9dy80000gn/X/F7776212-4447-5E1D-BADF-D23F26ACA647/d/Wrapper/ScoreImage.app' }, proceeding on the assumption it is not translocated: Error Domain=NSPOSIXErrorDomain Code=1 "Operation not permitted"
2023-06-02 16:31:03.698397+0800 ScoreImage[22728:1556762] [SceneConfiguration] Info.plist contained no UIScene configuration dictionary (looking for configuration named "(no name)")
2023-06-02 16:31:03.698677+0800 ScoreImage[22728:1556762] [SceneConfiguration] Info.plist contained no UIScene configuration dictionary (looking for configuration named "(no name)")
2023-06-02 16:31:03.698913+0800 ScoreImage[22728:1556762] [SceneConfiguration] Info.plist contained no UIScene configuration dictionary (looking for configuration named "(no name)")
[DEBUG] Start to load resnet50 Core ML Model
2023-06-02 16:31:03.917427+0800 ScoreImage[22728:1556762] [default] LSPrefs: could not find untranslocated node for <FSNode 0x600001ec80c0> { isDir = ?, path = '/private/var/folders/wk/frkkcch539lc6s2dk6dw9dy80000gn/X/F7776212-4447-5E1D-BADF-D23F26ACA647/d/Wrapper/ScoreImage.app' }, proceeding on the assumption it is not translocated: Error Domain=NSPOSIXErrorDomain Code=1 "Operation not permitted"
2023-06-02 16:31:04.009689+0800 ScoreImage[22728:1557198] Metal API Validation Enabled
2023-06-02 16:31:04.143203+0800 ScoreImage[22728:1556762] [SceneConfiguration] Info.plist contained no UIScene configuration dictionary (looking for configuration named "(no name)")
[DEBUG] resnet50 ML Model load success, cost 40682.12112500623 ms

[DEBUG] Start to load iqa Core ML Model
2023-06-02 16:37:19.636736+0800 ScoreImage[22728:1556762] [plugin] AddInstanceForFactory: No factory registered for id <CFUUID 0x60000188c7e0> 30010C1C-93BF-11D8-8B5B-000A95AF9C6A
2023-06-02 16:37:20.737048+0800 ScoreImage[22728:1556762] [] CMIO_DAL_CMIOExtension_Device.mm:347:Device legacy uuid isn't present, using new style uuid instead
2023-06-02 16:37:20.841749+0800 ScoreImage[22728:1562376] IOMFB: FigInstallVirtualDisplay not found
2023-06-02 16:37:20.841862+0800 ScoreImage[22728:1562376] iomfb_populate_display_infos: Local call to iomfb_match_callback
2023-06-02 16:37:20.842390+0800 ScoreImage[22728:1562376] iomfb_match_callback: primary
2023-06-02 16:37:20.842595+0800 ScoreImage[22728:1562376] iomfb_match_callback: external-0
2023-06-02 16:37:20.842653+0800 ScoreImage[22728:1562376] iomfb_match_callback: external-1
2023-06-02 16:37:20.906459+0800 ScoreImage[22728:1556762] [plugin] AddInstanceForFactory: No factory registered for id <CFUUID 0x60000189a720> F8BB1C28-BAE8-11D6-9C31-00039315CD46
2023-06-02 16:37:29.481427+0800 ScoreImage[22728:1562593] [] CMIO_DP_HardwarePlugInInterface.cpp:791:HardwarePlugIn_StreamCopyBufferQueue CMIO::DP::HardwarePlugIn_StreamCopyBufferQueue: no queue altered proc
2023-06-02 16:37:29.481533+0800 ScoreImage[22728:1562593] [] CMIO_DAL_PlugIn.cpp:371:StreamCopyBufferQueue Error: 1852797029, got an error from the plug-in routine
2023-06-02 16:37:29.482029+0800 ScoreImage[22728:1562593] [] CMIOHardware.cpp:1222:CMIOStreamCopyBufferQueue Error: 1852797029, failed
2023-06-02 16:37:29.482184+0800 ScoreImage[22728:1562593] [] CMIOHardware.cpp:1388:CMIOStreamRegisterAsyncStillCaptureCallback stream doesn't support async still capture
2023-06-02 16:37:29.482209+0800 ScoreImage[22728:1562593] [] CMIOHardware.cpp:1412:CMIOStreamRegisterAsyncStillCaptureCallback Error: 1970171760, failed
2023-06-02 16:37:29.997233+0800 ScoreImage[22728:1556762] +[CATransaction synchronize] called within transaction
2023-06-02 16:37:59.402366+0800 ScoreImage[22728:1556762] [API] cannot add handler to 3 from 3 - dropping
2023-06-02 16:38:07.707544+0800 ScoreImage[22728:1556762] [API] cannot add handler to 3 from 3 - dropping
2023-06-02 16:38:10.377848+0800 ScoreImage[22728:1556762] [API] cannot add handler to 3 from 3 - dropping
[DEBUG] iqa ML Model load success, cost 586356.1193750065 ms

```

**after change core ml model to core ml package, it cost almost 15 minutes to load `IQA_ML_Neural_Network.mlpackage`**

```
[DEBUG] iqa ML Model load success, cost 902217.6542916714 ms
```

## Error computing NN outputs

```
 2023-06-02 15:08:28.200445+0800 ScoreImage[15203:1456087] +[CATransaction synchronize] called within transaction
 2023-06-02 15:08:33.323865+0800 ScoreImage[15203:1463559] [espresso] [Espresso::handle_ex_plan] exception=Espresso exception: "Invalid state": reshape mismatching size: 2147483647 1 1 1 1 -> 32 24 384 1 1 status=-5
 2023-06-02 15:08:33.324679+0800 ScoreImage[15203:1463559] [coreml] Error computing NN outputs -5
 2023-06-02 15:08:33.325092+0800 ScoreImage[15203:1463559] [coreml] Failure in -executePlan:error:.
```

https://github.com/apple/coremltools/issues/412

I guess Upsamle / Resize has this issue when **usesCPUOnly=false**.

When i turn usesCPUOnly as true, no error occurs.

```
ML_EXPORT
@interface MLPredictionOptions : NSObject

/// Set to YES to force computation to be on the CPU only
@property (readwrite, nonatomic) BOOL usesCPUOnly API_DEPRECATED_WITH_REPLACEMENT("Use -[MLModelConfiguration computeUnits] instead.", macos(10.13, 12.0), ios(11.0, 15.0), tvos(11.0, 15.0), watchos(4.0, 8.0));
```

## MLComputeUnits

https://developer.apple.com/documentation/coreml/mlcomputeunits?language=objc

Use this enumeration to set or inspect the processing units you allow a model to use when it makes a prediction.
Use all to allow the OS to select the best processing unit to use (including the neural engine, if available).
Use MLComputeUnitsCPUOnly to restrict the model to the CPU, if your app might run in the background or runs other GPU intensive tasks.

## 5 Practical Ways to Speed Up your Deep Learning Model

https://blog.xmartlabs.com/2020/06/01/how-to-speed-up-inference-in-your-deep-learning-model/

Renzo GamboneJun 1, 2020
Did you obtain good accuracy with your deep learning model only to find out that inference time isn’t enough to deploy on production? Are you lost on how to optimize your model’s inference speed? Then this post is for you.


Data Science projects have the peculiarity that your focus must constantly shift across completely different requirements.

How will you obtain the data? Are you creating the dataset? (if so what exactly should you label? how many samples should you invest in?)
How will you make use of the model? A model is nice and fine, but how do you make a product that adds value to the user?
What model will you implement? There are many scientific papers out there, and often from the time you begin research until you are in the middle of implementation, the state of the art has already changed.
But the most important question of all. Is all of this viable? Can a model be trained using the data your budget allows? Will it perform sufficiently good for the use case we intend to apply it to?
And it is often that in this last question we focus mostly on obtaining the best accuracy for the dataset. This makes the most sense, as it allows us to validate if the project is feasible. If we need to invest in more data for the model to fulfill its goal.

For some projects, slow inference speed is not a dealbreaker. But what happens when it is?

This can happen to any deep learning project, but is often the case when deploying a project that involves object detection to production. When images from a camera are consumed each processed frame per second counts.

Investing more on hardware can definitely fix or alleviate our problem. But if we intend on selling our product as a solution (say a video camera with an integrated edge GPU) this linearly increases the cost of the product up to a point when we are not able to make returns from our investment.

Here at Xmartlabs we experienced this issue, researched what options we had available, and developed a checklist of sorts regarding different approaches we can take in order to improve inference time and how these can be combined.

Altering the model’s weights

The first step on optimizing our model runtime is to make the most out of our model’s architecture.

Post-training quantization

Converting the model’s weights from floating point (32-bits) to integers (8-bits) will degrade accuracy, but it significantly decreases model size in memory, while also improving CPU and hardware accelerator latency.

How to implement this approach will vary greatly depending on what framework is the model implemented on.

If the model is implemented on TensorFlow that’s great! Fortunately for you TensorFlows gives native support to model quantization on GPU.

If the model is implemented on PyTorch… Well, then it’s not that great. At the moment of writing this article PyTorch’s support to quantization is only on a CPU backend. Future PyTorch development aims to provide support for quantization on GPU, but at the time this is not the case in the stable version.

So in order to quantize a PyTorch model, it must be run on NVIDIA TensorRT, but for running it in TensorRT runtime the PyTorch model must be converted. As of now, the only way to achieve this conversion is to first convert the PyTorch model to ONNX, and then finally convert it to TensorRT.

Our experience

When we tried to quantize a PyTorch Faster R-CNN model we, unfortunately, run into multiple issues.

On paper, all this process makes sense and should be easy enough to do. However, in practice issues may occur from all these conversions. This is mainly due to the fact that the development of PyTorch, ONNX, and TensorRT goes in multiple directions when a feature is added into one an old integration won’t necessarily support it.

You may be able to run the model on ONNX, but issues may occur when converting ONNX to TensorRT, especially with some layers, such as the Resize layer in PyTorch.
At the time we tried to do this conversion, it happened to us that models built with PyTorch v1.3 or higher can be converted and run in ONNX Runtime, but can’t be run through the ONNX optimizer (which makes important improvements in the converted network).
Do keep in mind that these issues may or may not arise depending in the architecture of our model, we had no issue converting a simple CNN network, however with the Faster R-CNN implementation we were working with, that’s another story.

Some users have managed to solve their issues during conversion by downgrading PyTorch. However, this limits the ONNX opset you can access, which in turn also limits what TensorRT version will you be able to run your engine with.

Hopefully, all these issues will be solved in the near future… But considering that all these frameworks have high development speed it’s likely that there will always be short-lived incompatibility periods.

Post-training quantization is definitely a powerful tool, and although some PyTorch models can’t be quantized with this method, you should still give it a shot, considering that after exporting your model to ONNX it doesn’t take much effort trying to convert it via command line using trtexec, which by the way is readily available along with TensorRT in the Nvidia TensorRT docker container. If PyTorch quantization fails then we’d advise on looking for a TensorFlow implementation if quantization is still an approach you want to go for.

Converting the model to half-precision

In a similar fashion to the previous approach, this alternative aims to trade-off accuracy for speed and memory efficiency. It offers a middle point between FP32 and UInt8, where:

The model size is reduced by up to half (instead of by up to 75%)
The diminish of accuracy is less than UInt8, which brings the accuracy trade-off even more closer to FP32.
Most neural network weights already fall into this range, although doing this conversion risks gradient underflow (small gradient values becoming zeroes), which prevents the network from correctly learning anything.
Considering that nowadays the architecture of the GPUs is shifted to being optimized for FP16 operations, especially using tensor cores, this approach offers a great trade-off for increasing speed.

Moreover, it turns out that not all layers of the network take a significant time during inference. This means that we can find an even better trade-off by using half-precision only in the layers that need a speed boost (such as convolutions) and leave the rest in FP32. Even better, having some layers in FP32 helps preventing gradient underflow.

This approach is called Automatic Mixed Precision, and it differs on quantization in that rather than post-processing the weights of a trained model, the model should be trained, using mixed precision from the get-go.

Our experience

**TensorFlow once again goes great extents to make our lives easier by giving us native support to achieve this. Whereas if you’re using PyTorch, NVIDIA Apex is the tool you should use, which fortunately according to our experiences has way fewer issues than the pain that is doing quantization with a PyTorch model.**

https://www.tensorflow.org/api_docs/python/tf/train/experimental/enable_mixed_precision_graph_rewrite

https://github.com/NVIDIA/apex

Integrating Apex is advertised as a mere addition of three lines to your code. And actually, it is not much more than that. You must initialize it, change the backward pass call to use Apex’s scaled loss, and modify the way you save and load checkpoints.

from apex.fp16_utils import *
from apex import amp, optimizers
...
###  Initialization

opt_level = 'O1'
model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

###  Train your model
...
with amp.scale_loss(loss, optimizer) as scaled_loss:
    scaled_loss.backward()
...
The results are pretty good as well. How much speedup you get will strongly depend on the model you are training, but we got over 30% speed improvement without any impact on the accuracy of our Faster R-CNN model.

**Finding the best model**

We tried to extract every last drop out of our model in order to improve inference time. But maybe it wasn’t enough. Maybe our model’s architecture is simply too big for the problem we are attempting to solve.

Would diminishing the size of our model also decrement our accuracy? Not necessarily! This not only depends on the specific nature of the problem we want the model to solve. But also through research new model architectures are being constantly proposed and experimented with, often resulting in a slimmer architecture that through design achieves even better accuracy!

What’s even better is that if we implemented any of the previously described approaches we can then reuse that work with any modification we decide to make to our model.

**Changing the model’s backbone**

When doing transfer learning we can view our model’s backbone and the dataset on which it was pre-trained as a mere hyperparameter we are using in Grid Search.

We don’t need a fully trained model to evaluate inference time. This allows us to do experimentation with multiple backbones, and see which offer a better improvement to inference time.

We should expect a fair amount of improvement to our inference time, keeping in mind that our model still has to go through a backbone, and that although a fair amount of inference time happens at the backbone, the outer layers of our model can still have a big impact on inference time.

After researching which backbones offer better timings a model using this would need to be fully re-trained for us to analyze the backbone’s impact in the model’s accuracy.

**Changing the whole model**

A data scientist’s daily job is not only to work on data science projects but also on keeping an eye on research and how it affects the current state of the art.

Even though our model’s backbone is a big component of our model, we can only do so much by trying to optimize something keeping other things static.

If after all approaches inference time is still not of your liking, then it’s time to look at newly developed models and validate that what these models promise applies to your use case.

**Our experience**

Taking the problem of Object Detection as an example, some models are specifically optimized for speed such as YOLO, while at the same time other models offer multiple configurations that vary on the depth of the neural network and the size of the input they receive such as EfficentDet, allowing you to train and compare how the accuracy vs speed trade-off varies.

What’s more, the amazing machine learning community usually provides Open Source implementations of these models, for us to collaborate and help us not to reinvent the wheel! Such as this PyTorch implementation of EfficientDet by zylo117.

Knowledge Distillation

Our last proposed option to improve our model’s inference time is through knowledge distillation.

Say we have a big model (or an ensemble of models) which predicts with great accuracy, but its inference speed is undesirable.

Knowledge Distillation proposes to train a smaller model with fewer parameters by using our big model as trainer. This essentially trains our small model to output the same predictions as our big model or ensemble.

A great advantage of this is that we aren’t restricted to only using labelled data. Beware though that our accuracy might suffer a bit, but we should be able to get a decent speed improvement out of it.

We, unfortunately, didn’t have the pleasure of implementing this approach ourselves. But Knowledge distillation has got quite popular recently and has been used in object classification, object detection, acoustic models, and NLP, among others.

If you want to know more about knowledge distillation check out this paper by Geoffrey Hinton et al.

**To sum up**

In this blog, we’ve described five approaches to improve the inference time of your deep learning model. In particular, we’d advise you to implement them in the order we also listed them in, because any coding we do to implement model quantization and automatic mixed-precision is of great value to any further changes we make on our model.

We hope that this article was of value to you, either by giving you guidance on an issue you’re currently facing or by arming you with our knowledge should the need arise for it!

Let us know if you have an approach to speed up your model’s inference that wasn’t covered in this blogpost… Have any question about model’s inference speed optimization? We’d be happy to answer those in the comments if we can.

If you liked this blog post, We’ve covered more mobile ML topics such as How to convert a NN model from TensorFlow Lite to CoreML and Latest updates on apple machine learning.

## NVIDIA/apex
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Reshape in Pytorch leads to "Error computing NN outputs."

https://github.com/apple/coremltools/issues/1461

That is not a very helpful error message. **Fortunately, converting to an mlprogram (i.e. adding convert_to='mlprogram' to the ct.convert call) gives a better error message**:

> compiler error:  Encountered an error while compiling a neural network model: in operation of type reshape: Rank of the shape parameter must be between 0 and 5 (inclusive) in reshape".

The Core ML Framework can not handle tensors with a rank greater than five.

Since this issue can not be fixed in the coremltools repository, I'm going to close this issue.

@antoleb - If you would like to request that the Core ML Framework supports tensors with a rank greater than five or that the compiler gives a better error message for the NNv1 backend, please do that using the Feedback Assistant for Developers.

https://developer.apple.com/bug-reporting/

// convert_IQA_ML_script.py
```
        # As an alternative, you can convert the model to a neural network by eliminating the convert_to parameter:
        core_ml_neural_network_model = ct.convert(
                                                  traced_script_module,
                                                  convert_to="mlprogram",
                                                  inputs=[
                                                          ct.TensorType(name="mask_inputs", shape=mask_inputs.shape),  # Replace with the correct shape
                                                          ct.TensorType(name="feat_dis_org", shape=feat_dis_org.shape),  # Replace with the correct shape
                                                          ct.TensorType(name="feat_dis_scale_1", shape=feat_dis_scale_1.shape),  # Replace with the correct shape
                                                          ct.TensorType(name="feat_dis_scale_2", shape=feat_dis_scale_2.shape),  # Replace with the correct shape
                                                          ]
                                                  )
```

```
(torch-gpu) ➜  MUSIQ-Pytorch git:(main) ✗ python3 convert_IQA_ML_script.py 
Torch version 2.1.0.dev20230422 has not been tested with coremltools. You may run into unexpected errors. Torch 2.0.0 is the most recent version that has been tested.
```

## How to quantization pytorch model by NVIDIA Apex in Mac

To quantize a PyTorch model using NVIDIA Apex on a Mac, you can follow these steps:

1. Install NVIDIA Apex by running the following command in your terminal:

`pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex.git`

2. Import the necessary modules in your PyTorch script:

```
import torch
from apex import amp
```

3. Load your PyTorch model and define your loss function and optimizer as usual.

4. Wrap your model and optimizer with Apex's amp.initialize function:

`model, optimizer = amp.initialize(model, optimizer, opt_level="O2")`

5. Train your model as usual, but make sure to use the amp.scale_loss function to scale your loss before backpropagation:

```
with amp.scale_loss(loss, optimizer) as scaled_loss:
    scaled_loss.backward()
```

6. After training, you can quantize your model using Apex's amp.initialize function again, but this time with the opt_level argument set to "O1":
```
model = amp.initialize(model, opt_level="O1")
```

7. Save your quantized model as usual.

**Note that quantization may result in a loss of accuracy, so it's important to evaluate your quantized model's performance before deploying it!**

## What is QoS (Quality of Service) in GCD? - Swift

https://www.swiftpal.io/articles/what-is-qos-quality-of-service-in-gcd-swift

QoS can be used with .async() function, and the priorities are divided into four main categories.
    •    userInteractive: Used for animations, or updating UI.
    •    userInitiated: Used for tasks like loading data from API, preventing the user from making interactions.
    •    utility: Used for tasks that do not need to be tracked by the user.
    •    background: Used for tasks like saving data in the local database or any maintenance code which is not on high priority.

## core ml force MLModelConfiguration to use GPU not cpu

To force MLModelConfiguration to use the GPU instead of the CPU, you can set the computeUnits property of the configuration to .all. This will allow the model to use all available compute units, including the GPU. Here's an example of how to do this:

```
import CoreML

let configuration = MLModelConfiguration()
configuration.computeUnits = .all

let model = try! MyModel(configuration: configuration)
```

Replace MyModel with the name of your model class. This should allow your model to use the GPU for inference 

## Reducing the Size of Your Core ML App

https://developer.apple.com/documentation/coreml/model_customization/reducing_the_size_of_your_core_ml_app?language=objc

## Downloading and Compiling a Model on the User’s Device

https://developer.apple.com/documentation/coreml/downloading_and_compiling_a_model_on_the_user_s_device?language=objc

## Compressing ML Program Weights

https://coremltools.readme.io/docs/compressing-ml-program-weights

## Compressing Neural Network Weights

https://coremltools.readme.io/docs/quantization

## core ml use .compression_utils to compress program model from pytorch

To compress a PyTorch model for use with Core ML, you can use the .compression_utils module provided by Core ML. Here's an example of how to use it:
```
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
from torchvision import models
import coremltools as ct

# Load the PyTorch model
model = models.resnet18(pretrained=True)

# Optimize the model for mobile deployment
model.eval()
model = optimize_for_mobile(model)

# Convert the model to Core ML format
mlmodel = ct.convert(
    model,
    inputs=[ct.ImageType(name='input', shape=model.input_shape)],
    output_names=['output'],
)

# Compress the model using Core ML's compression_utils
compressed_model = ct.compression_utils.compress(
    mlmodel,
    ct.compression_utils.ModelType.MLPROGRAM,
    ct.compression_utils.CompressionAlgorithm.LZ4,
)

# Save the compressed model to a file
compressed_model.save('compressed.mlmodel')
```

In this example, we first load a pre-trained ResNet18 model from PyTorch. We then optimize the model for mobile deployment using optimize_for_mobile. Next, we convert the model to Core ML format using ct.convert. Finally, we compress the model using ct.compression_utils.compress and save it to a file.

Note that the compress function takes three arguments: the Core ML model to compress, the type of model (either MLPROGRAM or NEURALNETWORK), and the compression algorithm to use (either LZ4 or QUANTIZATION). In this example, we're using MLPROGRAM and LZ4, but you can experiment with different options to see what works best for your model 

## different between LZ4 and QUANTIZATION

LZ4 and QUANTIZATION are two different concepts in Core ML compression.

LZ4 is a compression algorithm that can be used to reduce the size of Core ML models. It is a lossless compression algorithm, which means that the compressed model can be decompressed back to its original size without any loss of information. LZ4 is generally faster than other algorithms like ZLIB or LZFSE.

QUANTIZATION, on the other hand, is a technique used to reduce the size of neural network models by reducing the precision of the weights and activations. This can result in a significant reduction in model size, but it can also lead to a loss of accuracy. There are several types of quantization, including post-training quantization, which quantizes the weights and activations after training, and quantization-aware training, which trains the model with quantization in mind.

When you use ct.compression_utils.compress to compress a Core ML model, you need to specify both the type of model (MLPROGRAM or NEURALNETWORK) and the compression algorithm (LZ4, ZLIB, or LZFSE). In the example code I provided earlier, we used MLPROGRAM and LZ4 to compress a PyTorch model for use with Core ML. However, you can experiment with different options to see what works best for your specific use case. If you're concerned about model accuracy, you may want to consider using quantization instead of or in addition to compression 

## After compress core ml package...
```
[DEBUG] iqa ML Model load success, cost 883764.8360416642 ms
```
Memory Highest Usage: 3.12GB

## Optimize your Core ML usage

Learn how Core ML works with the CPU, GPU, and Neural Engine to power on-device, privacy-preserving machine learning experiences for your apps. We'll explore the latest tools for understanding and maximizing the performance of your models. We'll also show you how to generate reports to easily understand your model performance characteristics, help you gain insight into your models with the Core ML Instrument, and take you through API enhancements to further optimize Core ML integration in your apps. To get the most out of this session, be sure to watch “Tune your Core ML models” from WWDC21.

https://developer.apple.com/videos/play/wwdc2022/10027/

> Train model for Core ML

> Performance Report for Core ML Package/Model

> Compress Core ML Package/Model

## core ml train model from dataset SPAQ

To train a Core ML model from a dataset, you will need to follow a few steps. First, you will need to prepare your dataset by organizing it into the appropriate format for training. This typically involves splitting your data into training and validation sets, and then converting your data into a format that can be read by Core ML.

Once your dataset is prepared, you can use a machine learning framework such as TensorFlow or PyTorch to train your model. These frameworks provide a wide range of tools and algorithms for training machine learning models, and can be used to train models for a variety of tasks.

After you have trained your model, you can use Core ML to convert it into a format that can be used on iOS devices. Core ML provides a set of tools and APIs for working with machine learning models, and can be used to integrate your model into your iOS app.

Overall, training a Core ML model from a dataset requires a combination of machine learning expertise and knowledge of the Core ML framework. With the right tools and techniques, however, it is possible to create powerful machine learning models that can be used to enhance the functionality of your iOS apps 

## Creating a Model from Tabular Data

https://developer.apple.com/documentation/createml/creating_a_model_from_tabular_data

## Example of speeding up inference of PyTorch models on M1 via Core ML tools

https://drsleep.github.io/technical/Neural-Sketching-CoreML/

## CoreML of ResNet50 for iOS app

https://iosexample.com/coreml-of-resnet50-for-ios-app/

https://github.com/madara-tribe?tab=repositories

## Converting ONNX Image Classification Models to Core ML

https://www.codeproject.com/Articles/5277526/Converting-ONNX-Image-Classification-Models-to-Cor

## AI Image Classification on iOS with ResNet

https://www.codeproject.com/Articles/5277527/AI-Image-Classification-on-iOS-with-ResNet

## Apple - Core ML Models

https://developer.apple.com/machine-learning/models/

https://ml-assets.apple.com/coreml/models/Image/ImageClassification/Resnet50/Resnet50.mlmodel

## Classifying Images with Vision and Core ML

https://developer.apple.com/documentation/vision/classifying_images_with_vision_and_core_ml?language=objc
