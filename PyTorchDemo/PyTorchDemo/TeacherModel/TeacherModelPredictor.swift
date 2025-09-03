import UIKit

class TeacherModelPredictor: Predictor {
    private var isRunning: Bool = false
    private lazy var module: TeacherTorchModule = {
        if let filePath = Bundle.main.path(forResource: "teacher_model", ofType: "ptl"),
           let module = TeacherTorchModule(fileAtPath: filePath) {
            return module
        } else {
            fatalError("Failed to load model!")
        }
    }()
    
    func predict(_ image: UIImage) throws -> ([(String, Float)], Double)? {
        if isRunning {
            return nil
        }
        isRunning = true
        let startTime = CACurrentMediaTime()
        
        // Resize image to match model input size (1188x1914)
        let scaledImageSize = CGSizeMake(CGFloat(TeacherModelConstants.inputImageWidth), CGFloat(TeacherModelConstants.inputImageHeight))
        guard let resizedImage = image.scaledImage(with: scaledImageSize) else {
            return nil
        }
        
        // Convert to pixel buffer
        guard let pixelBuffer = resizedImage.pixelBuffer() else {
            return nil
        }
        
        // Normalize pixel values using MobileIQA-specific normalization
        guard let normalizedBuffer = pixelBuffer.normalizedForMobileIQA(Int(scaledImageSize.width), Int(scaledImageSize.height)) else {
            return nil
        }
        var tensorBuffer = normalizedBuffer
        
        // Run prediction
        let scores: [(String, Float)] = module.predict(image: UnsafeMutableRawPointer(&tensorBuffer), 
                                                     size: scaledImageSize).map { array in
            guard let imageName = array[0] as? String,
                  let score = array[1] as? NSNumber else {
                return ("unknown", 0.0)
            }
            return (imageName, score.floatValue)
        }
        
        isRunning = false
        let inferenceTime = CACurrentMediaTime() - startTime
        return (scores, inferenceTime)
    }
}

// Constants for the model
public enum TeacherModelConstants {
    // MobileIQA model settings
    static let modelType: String = "mobileIQA"
    // Reduced size while maintaining aspect ratio (1907:1231 ≈ 1.55)
    static let inputImageWidth: Int = 636  // 1907/3 for memory efficiency
    static let inputImageHeight: Int = 410  // 1231/3 for memory efficiency
    
    // Normalization parameters for MobileIQA
    static let normalizationMean: [Float] = [0.485, 0.456, 0.406]
    static let normalizationStd: [Float] = [0.229, 0.224, 0.225]
}
