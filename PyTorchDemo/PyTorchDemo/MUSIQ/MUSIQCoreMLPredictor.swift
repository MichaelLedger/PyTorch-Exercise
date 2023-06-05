//
//  MUSIQCoreMLPredictor.swift
//  PyTorchDemo
//
//  Created by Gavin Xiang on 2023/6/1.
//

import UIKit
import CoreML
import Vision

class MUSIQCoreMLPredictor {
    private var loaded: Bool = false
    
    private var isRunning: Bool = false
    
    fileprivate var resnet50MLModel: resnet50_ML_Neural_Network?
    
    //'IQA_ML_Program_Compressed' is only available in iOS 16.0 or newer
    fileprivate var iqaMLModel: IQA_ML_Program_Compressed?
    
    private var activateIQA: Bool = true
    
    // Call can throw, but errors cannot be thrown out of a property initializer
    //    lazy var musiqMLModel: IQA_ML_Program_Compressed? = {
    //        do {
    //            let model = try IQA_ML_Program_Compressed()
    //            return model
    //        } catch {
    //            // Handle the error
    //            print("Error initializing the model: \(error)")
    //            return nil
    //        }
    //    }()
    
    //    var musiqVNCoreMLModel: VNCoreMLModel
    
    //    var resnet50VNCoreMLModel: VNCoreMLModel?
    
    /*
     typedef NS_ENUM(NSInteger, MLComputeUnits) {
         MLComputeUnitsCPUOnly = 0,
         MLComputeUnitsCPUAndGPU = 1,
         MLComputeUnitsAll = 2,
         MLComputeUnitsCPUAndNeuralEngine API_AVAILABLE(macos(13.0), ios(16.0), watchos(9.0), tvos(16.0)) = 3
     } API_AVAILABLE(macos(10.14), ios(12.0), watchos(5.0), tvos(12.0));
     */
    
    init() {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            
            var startTime = CACurrentMediaTime()
            print("[DEBUG] Start to load resnet50 Core ML Model")
            
            let resnet50Config = MLModelConfiguration()
            resnet50Config.computeUnits = .all //.cpuOnly
            
            guard let resnet50 = try? resnet50_ML_Neural_Network(configuration: resnet50Config) else {
                fatalError("Failed to load the resnet50 Core ML model.")
            }
            self?.resnet50MLModel = resnet50
            
            var loadTime = (CACurrentMediaTime() - startTime) * 1000
            print("[DEBUG] resnet50 ML Model load success, cost \(loadTime) ms")
            
            if let self = self, self.activateIQA {
                startTime = CACurrentMediaTime()
                print("[DEBUG] Start to load iqa Core ML Model")
                
                /*
                 If a large core ML model is loaded in the main thread, it will definitely crash!
                 The app “ScoreImage” on Gavin’s iPhone XS Max quit unexpectedly.
                 Domain: IDEDebugSessionErrorDomain
                 Code: 11
                 Failure Reason: Message from debugger: Terminated due to memory issue
                 */
                let iqaConfig = MLModelConfiguration()
                iqaConfig.computeUnits = .all //.cpuOnly
                guard let iqa = try? IQA_ML_Program_Compressed(configuration: iqaConfig) else {
                    fatalError("Failed to load the iqa Core ML model.")
                }
                self.iqaMLModel = iqa
                
                loadTime = (CACurrentMediaTime() - startTime) * 1000
                print("[DEBUG] iqa ML Model load success, cost \(loadTime) ms")
            }
            
            self?.loaded = true
        }
        
        //        guard let musiqModelURL = Bundle.main.url(forResource: "IQA_ML_Program_Compressed", withExtension: "mlmodel") else {
        //            fatalError("Failed to find the model file.")
        //        }
        //        guard let musiqModel = try? VNCoreMLModel(for: MLModel(contentsOf: musiqModelURL)) else {
        //            fatalError("Failed to load the Core ML model.")
        //        }
        //        guard let model = musiqMLModel?.model, let musiqModel = try? VNCoreMLModel(for: model) else {
        //            fatalError("Failed to load the Core ML model.")
        //        }
        //        musiqVNCoreMLModel = musiqModel
        //
        //        guard let resnet50ModelURL = Bundle.main.url(forResource: "resnet50_ML_Neural_Network", withExtension: "mlmodelc") else {
        //            fatalError("Failed to find the model file.")
        //        }
        //        guard let resnet50Model = try? VNCoreMLModel(for: MLModel(contentsOf: resnet50ModelURL)) else {
        //            fatalError("Failed to load the Core ML model.")
        //        }
        //        guard let resnet50Model = try? VNCoreMLModel(for: resnet50MLModel.model) else {
        //            fatalError("Failed to create VNCoreMLModel")
        //        }
        //        resnet50VNCoreMLModel = resnet50Model
        
        //        resnet50VNCoreMLModel = loadResNet50Model()
    }
    
    /*
     old input shape:
     MultiArray (Float32 1 × 3 × 224 × 224)
     
     output:
     MultiArray (Float32 1 × 2048 × 24 × 32)
     MultiArray (Float32 1 × 2048 × 9 × 12)
     MultiArray (Float32 1 × 2048 × 5 × 7)
     
     width/height
     32/24: 1024 * 768 = 786432, 786432 * 2 = 1572864
     12/9: 384 * 288 = 110592, 110592 * 2 = 221184
     7/5: 224 * 160 = 35840, 35840 * 2 = 71680
     */
    func preprocess(image: UIImage, imageSize: CGSize, shapeSize: CGSize) -> MLMultiArray? {
        let imageSize = CGSize(width: 224, height: 224)
//        guard let pixels = image.resize(to: imageSize).pixelData()?.map({ (Double($0) / 255.0 - 0.5) * 2 }) else {
//            return nil
//        }
//
//        let r = pixels.enumerated().filter { $0.offset % 4 == 0 }.map { $0.element }
//        let g = pixels.enumerated().filter { $0.offset % 4 == 1 }.map { $0.element }
//        let b = pixels.enumerated().filter { $0.offset % 4 == 2 }.map { $0.element }
//
        /*
         32/24: 1024 * 768
         Thread 2: "Could not store NSNumber at offset 1572864 because it is beyond the end of the multi array."
         */
//        let combination = r + g + b
//        for (index, element) in combination.enumerated() {
//            array[index] = NSNumber(value: element)
//        }
        
        guard let resizedImage = image.scaledImage(with: imageSize) else {
            return nil
        }
        // original image
        guard let pixelBuffer = resizedImage.pixelBuffer() else {
            return nil
        }
        guard let normalizedBuffer = pixelBuffer.normalized(Int(imageSize.width), Int(imageSize.height)) else {
            return nil
        }
        
        /*
         2023-06-02 13:19:03.673051+0800 ScoreImage[7946:1347588] [coreml] Error Domain=com.apple.CoreML Code=1 "For input feature 'x', the provided shape 1 × 3 × 768 × 1024 is not compatible with the model's feature description." UserInfo={NSLocalizedDescription=For input feature 'x', the provided shape 1 × 3 × 768 × 1024 is not compatible with the model's feature description., NSUnderlyingError=0x600001650d80 {Error Domain=com.apple.CoreML Code=0 "MultiArray shape (1 x 3 x 768 x 1024) does not match the shape (1 x 3 x 224 x 224) specified in the model description" UserInfo={NSLocalizedDescription=MultiArray shape (1 x 3 x 768 x 1024) does not match the shape (1 x 3 x 224 x 224) specified in the model description}}}
         2023-06-02 13:19:03.673267+0800 ScoreImage[7946:1347588] [coreml] Failure verifying inputs.
         */
//        guard let array = try? MLMultiArray(shape: [1, 2048, shapeSize.height as NSNumber, shapeSize.width as NSNumber], dataType: .double) else {
//            return nil
//        }
//        guard let array = try? MLMultiArray(shape: [1, 3, imageSize.height as NSNumber, imageSize.width as NSNumber], dataType: .double) else {
//            return nil
//        }
        guard let array = try? MLMultiArray(shape: [1, 3, 224, 224], dataType: .double) else {
            return nil
        }
        // (lldb) po normalizedBuffer.count
        // 2359296
        for (index, element) in normalizedBuffer.enumerated() {
            array[index] = NSNumber(value: element)
        }
        
        return array
    }
    
//    func preprocess(image: UIImage, size: CGSize) -> MLMultiArray? {
//        //        guard let resizedImage = image.scaledImage(with: size) else {
//        //            return nil
//        //        }
//        //        // original image
//        //        guard let pixelBuffer = resizedImage.pixelBuffer() else {
//        //            return nil
//        //        }
//        //        guard let normalizedBuffer = pixelBuffer.normalized(Int(size.width), Int(size.height)) else {
//        //            return nil
//        //        }
//        guard let pixels = image.resize(to: size).pixelData()?.map({ (Double($0) / 255.0 - 0.5) * 2 }) else {
//            return nil
//        }
//        guard let array = try? MLMultiArray(shape: [1, 3, 224, 224], dataType: .double) else {
//            return nil
//        }
//
//        /*
//         *** Terminating app due to uncaught exception 'NSInvalidArgumentException', reason: 'Could not store NSNumber at offset 150528 because it is beyond the end of the multi array.'
//         */
//        //        for (index, element) in normalizedBuffer.enumerated() {
//        //            array[index] = NSNumber(value: element)
//        //        }
//
//        let r = pixels.enumerated().filter { $0.offset % 4 == 0 }.map { $0.element }
//        let g = pixels.enumerated().filter { $0.offset % 4 == 1 }.map { $0.element }
//        let b = pixels.enumerated().filter { $0.offset % 4 == 2 }.map { $0.element }
//
//        let combination = r + g + b
//        for (index, element) in combination.enumerated() {
//            array[index] = NSNumber(value: element)
//        }
//
//        return array
//    }
    
    
    /*
     2023-06-01 14:51:41.171766+0800 ScoreImage[714:90184] Metal API Validation Enabled
     Error loading Resnet50 model: Error Domain=com.apple.vis Code=15 "The model does not have a valid input feature of type image" UserInfo={NSLocalizedDescription=The model does not have a valid input feature of type image}
     */
    func loadResNet50Model() -> VNCoreMLModel? {
        guard let modelURL = Bundle.main.url(forResource: "resnet50_ML_Neural_Network", withExtension: "mlmodelc") else {
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
    
    func resnet50Predict(image: UIImage?, imageSize: CGSize, shapeSize: CGSize) -> MLMultiArray? {
        guard let image = image else {
            fatalError("Failed to load image.")
        }
        
        // old input shape: MultiArray (Float32 1 × 3 × 224 × 224)
//        guard let inputArray = preprocess(image: image, imageSize: imageSize, shapeSize: shapeSize) else {
//            fatalError("Failed to create input.")
//        }
//
//        let input = resnet50_ML_Neural_NetworkInput.init(x: inputArray)
        
        // new input shape: Image (Color 224 × 224)
        /// x as color (kCVPixelFormatType_32BGRA) image buffer, 224 pixels wide by 224 pixels high
        guard let resizedImage = image.scaledImage(with: imageSize) else {
            fatalError("Failed to scale image.")
        }
        guard let pixelBuffer = resizedImage.pixelBuffer() else {
            fatalError("Failed to generate pixel buffer.")
        }
//        let input = resnet50_ML_Neural_NetworkInput.init(input_image: pixelBuffer)
        
        guard let normalizedBuffer = pixelBuffer.normalized(Int(imageSize.width), Int(imageSize.height)) else {
            return nil
        }
        guard let array = try? MLMultiArray(shape: [1, 3, imageSize.height as NSNumber, imageSize.width as NSNumber], dataType: .double) else {
            return nil
        }
        // (lldb) po normalizedBuffer.count
        // 2359296
        for (index, element) in normalizedBuffer.enumerated() {
            array[index] = NSNumber(value: element)
        }
        
        let input = resnet50_ML_Neural_NetworkInput.init(input_image: array)
        
        /*
         2023-06-02 13:50:02.049864+0800 ScoreImage[10041:1378281] [coreml] Error Domain=com.apple.CoreML Code=1 "Input image feature x does not match model description" UserInfo={NSLocalizedDescription=Input image feature x does not match model description, NSUnderlyingError=0x6000021e42a0 {Error Domain=com.apple.CoreML Code=0 "Image size 1024 x 768 not in allowed set of image sizes" UserInfo={NSLocalizedDescription=Image size 1024 x 768 not in allowed set of image sizes}}}
         */
        
        // Make a prediction using the model
        guard let model = resnet50MLModel, let output = try? model.prediction(input: input) else {
            fatalError("Failed to make prediction.")
        }
        
        // Extract the prediction result from the output
        guard let var_830 = output.featureValue(for: "var_830"), let var_830_arrray: MLMultiArray = var_830.multiArrayValue else {
            fatalError("Failed to get prediction result.")
        }
        
        /*
         resnet50Predict==:Float32 1 × 2048 × 24 × 32 array
         resnet50Predict==:Float32 1 × 2048 × 9 × 12 array
         resnet50Predict==:Float32 1 × 2048 × 5 × 7 array
         */
        print("resnet50Predict==:\(var_830_arrray)")
        return var_830_arrray
    }
    
    func iqaPredict(mask_inputs: MLMultiArray, feat_dis_org: MLMultiArray, feat_dis_scale_1: MLMultiArray, feat_dis_scale_2: MLMultiArray) throws -> MLMultiArray? {
        let input = IQA_ML_Program_CompressedInput.init(mask_inputs: mask_inputs, feat_dis_org: feat_dis_org, feat_dis_scale_1: feat_dis_scale_1, feat_dis_scale_2: feat_dis_scale_2)
        
        /*
         2023-06-02 15:08:28.200445+0800 ScoreImage[15203:1456087] +[CATransaction synchronize] called within transaction
         2023-06-02 15:08:33.323865+0800 ScoreImage[15203:1463559] [espresso] [Espresso::handle_ex_plan] exception=Espresso exception: "Invalid state": reshape mismatching size: 2147483647 1 1 1 1 -> 32 24 384 1 1 status=-5
         2023-06-02 15:08:33.324679+0800 ScoreImage[15203:1463559] [coreml] Error computing NN outputs -5
         2023-06-02 15:08:33.325092+0800 ScoreImage[15203:1463559] [coreml] Failure in -executePlan:error:.
         */
        
        // prediction options
//        var predictionOpt = MLPredictionOptions()
//        predictionOpt.usesCPUOnly = true
        
        // Make a prediction using the model
        guard let model = iqaMLModel else {
            print("Failed to load iqa model.")
            return nil
        }
        do {
//            let output: IQA_ML_Program_CompressedOutput = try! model.prediction(input: input)
            let output: IQA_ML_Program_CompressedOutput = try model.prediction(input: input)
            // Extract the prediction result from the output
            let var_9627 = output.var_9627
            print("iqaPredict==:\(var_9627)")
            return var_9627
        } catch {
            throw error
        }
    }
    
    // completion(score, time_cost) // score range: 1~5
    func predict(image: UIImage, completion: @escaping(Float, Double) -> Void) {
        if !loaded {
            DispatchQueue.main.async {
                let alert = UIAlertController(title: "Alert", message: "Sorry, core ML package is still loading, please try it later.", preferredStyle: .alert)
                let ok = UIAlertAction(title: "OK", style: .default)
                alert.addAction(ok)
                UIApplication.shared.findKeyWindow()?.rootViewController?.present(alert, animated: true)
            }
            
            isRunning = false
            completion(1, 0)
            return
        }
        if isRunning {
            DispatchQueue.main.async {
                let alert = UIAlertController(title: "Alert", message: "Sorry, predict process is still running, please try it later.", preferredStyle: .alert)
                let ok = UIAlertAction(title: "OK", style: .default)
                alert.addAction(ok)
                UIApplication.shared.findKeyWindow()?.rootViewController?.present(alert, animated: true)
            }
            return
        }
        
        isRunning = true
        
        let startTime = CACurrentMediaTime()
        
        let scaledImageSize = CGSizeMake(CGFloat(MUSIQConstants.inputImageWidth), CGFloat(MUSIQConstants.inputImageHeight))
        let scaledImageSize1 = CGSizeMake(CGFloat(MUSIQConstants.inputImageWidth1), CGFloat(MUSIQConstants.inputImageHeight1))
        let scaledImageSize2 = CGSizeMake(CGFloat(MUSIQConstants.inputImageWidth2), CGFloat(MUSIQConstants.inputImageHeight2))
        
        // Create a dictionary with the input parameters
        //        let inputParameters: [String: Any] = [
        //            "x": pixelBuffer
        //        ]
        
        // Create a prediction input using the input parameters
        //        guard let input = try? MLDictionaryFeatureProvider(dictionary: inputParameters) else {
        //            fatalError("Failed to create input.")
        //        }
        //        let shape = try! MLMultiArray(shape: [1, 2, 3], dataType: .float32)
        //        guard let inputArray = preprocess(image: image) else {
        
        /*
         MultiArray (Float32 1 × 2048 × 24 × 32)
         MultiArray (Float32 1 × 2048 × 9 × 12)
         MultiArray (Float32 1 × 2048 × 5 × 7)
         
         width/height
         32/24: 1024 * 768 = 786432, 786432 * 2 = 1572864
         12/9: 384 * 288 = 110592, 110592 * 2 = 221184
         7/5: 224 * 160 = 35840, 35840 * 2 = 71680
         */
        let var_830_arrray = resnet50Predict(image: image.scaledImage(with: scaledImageSize), imageSize: scaledImageSize, shapeSize: CGSizeMake(32, 24))
        let var_830_arrray_1 = resnet50Predict(image: image.scaledImage(with: scaledImageSize1), imageSize: scaledImageSize1, shapeSize: CGSizeMake(12, 9))
        let var_830_arrray_2 = resnet50Predict(image: image.scaledImage(with: scaledImageSize2), imageSize: scaledImageSize2, shapeSize: CGSizeMake(7, 5))
        
        let maskInput = try! MLMultiArray(shape: [1, 912], dataType: .float32)
        let value: Float = 1.0
        for i in 0..<maskInput.count {
            maskInput[i] = value as NSNumber
        }
        
        guard let var_830_arrray = var_830_arrray, let var_830_arrray_1 = var_830_arrray_1, let var_830_arrray_2 = var_830_arrray_2 else {
            return
        }
        
        /*
         Float32 1 × 2048 × 7 × 7 array
         Float32 1 × 2048 × 7 × 7 array
         Float32 1 × 2048 × 7 × 7 array
         
         2023-06-02 11:59:34.295950+0800 ScoreImage[2318:1265897] [coreml] Error Domain=com.apple.CoreML Code=1 "For input feature 'feat_dis_scale_1', the provided shape 1 × 2048 × 7 × 7 is not compatible with the model's feature description." UserInfo={NSLocalizedDescription=For input feature 'feat_dis_scale_1', the provided shape 1 × 2048 × 7 × 7 is not compatible with the model's feature description., NSUnderlyingError=0x600000829110 {Error Domain=com.apple.CoreML Code=0 "MultiArray shape (1 x 2048 x 7 x 7) does not match the shape (1 x 2048 x 9 x 12) specified in the model description" UserInfo={NSLocalizedDescription=MultiArray shape (1 x 2048 x 7 x 7) does not match the shape (1 x 2048 x 9 x 12) specified in the model description}}}
         2023-06-02 11:59:34.296031+0800 ScoreImage[2318:1265897] [coreml] Failure verifying inputs.
         */
        if activateIQA {
            do {
                let finalResult = try iqaPredict(mask_inputs: maskInput, feat_dis_org: var_830_arrray, feat_dis_scale_1: var_830_arrray_1, feat_dis_scale_2: var_830_arrray_2)
                print(finalResult as Any)
                if let finalResult = finalResult {
                    let score = finalResult[0].floatValue
                    print("score==\(score)")
                    isRunning = false
                    let inferenceTime = (CACurrentMediaTime() - startTime) * 1000
                    completion(score, inferenceTime)
                }
            } catch let e {
                print("iqaPredict failed: \(e.localizedDescription)")
                isRunning = false
                let inferenceTime = (CACurrentMediaTime() - startTime) * 1000
                completion(1, inferenceTime)
            }
        } else {
            isRunning = false
            let inferenceTime = (CACurrentMediaTime() - startTime) * 1000
            completion(1, inferenceTime)
        }
        
        /*
         let request = VNCoreMLRequest(model: resnet50VNCoreMLModel) { [weak self] (request, error) in
         guard let results = request.results as? [VNClassificationObservation] else {
         fatalError("Failed to process the request.")
         }
         
         // Handle the results
         print(results)
         
         self?.isRunning = false
         //        let inferenceTime = (CACurrentMediaTime() - startTime) * 1000
         let inferenceTime = CACurrentMediaTime() - startTime
         //        let results = topK(scores: outputs, labels: labels, count: resultCount)
         //        guard let cgImage = postprocessImageData(numbers: outputs) else {
         //            return (results, inferenceTime)
         //        }
         //        let outputImage = UIImage(cgImage: cgImage) //test
         //        UIImageWriteToSavedPhotosAlbum(outputImage, nil, nil, nil)
         //        print("[DEBUG] SavedPhotosAlbum:\(outputImage)")
         
         //test
         completion(1, inferenceTime)
         }
         
         let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
         do {
         try handler.perform([request])
         } catch {
         print("Failed to perform the request: \(error.localizedDescription)")
         }
         */
        
        //        // Create a dictionary with the input parameters
        //        let inputParameters: [String: Any] = [
        //            "param1": param1,
        //            "param2": param2
        //        ]
        //
        //        // Create a prediction input using the input parameters
        //        guard let input = try? MLDictionaryFeatureProvider(dictionary: inputParameters) else {
        //            fatalError("Failed to create input.")
        //        }
        //
        //        // Make a prediction using the model
        //        guard let output = try? model.prediction(from: input) else {
        //            fatalError("Failed to make prediction.")
        //        }
        //
        //        // Extract the prediction result from the output
        //        guard let result = output.featureValue(for: "result")?.stringValue else {
        //            fatalError("Failed to get prediction result.")
        //        }
        
    }
}
