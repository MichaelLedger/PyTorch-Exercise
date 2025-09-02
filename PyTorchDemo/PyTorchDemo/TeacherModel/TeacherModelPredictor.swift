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
    
    func predict(_ image: UIImage) throws -> (Float, Double)? {
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
        
        // Normalize pixel values
        guard let normalizedBuffer = pixelBuffer.normalized(Int(scaledImageSize.width), Int(scaledImageSize.height)) else {
            return nil
        }
        var tensorBuffer = normalizedBuffer
        
        // Run prediction
        let score = module.predict(image: UnsafeMutableRawPointer(&tensorBuffer), 
                                 size: scaledImageSize)
        
        isRunning = false
        let inferenceTime = CACurrentMediaTime() - startTime
        return (score, inferenceTime)
    }
}

// Constants for the model
public enum TeacherModelConstants {
    static let inputImageWidth: Int = 1914  // From the log
    static let inputImageHeight: Int = 1188 // From the log
}