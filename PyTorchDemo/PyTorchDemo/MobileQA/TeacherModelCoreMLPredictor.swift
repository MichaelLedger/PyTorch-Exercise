import UIKit
import CoreML

/// A predictor that uses the Core ML version of the TeacherModel
class TeacherModelCoreMLPredictor {
    private var isRunning: Bool = false
    
    /// The Core ML model instance
    private lazy var model: TeacherModel = {
        let config = MLModelConfiguration()
        config.computeUnits = .all // Use all available compute units (CPU, GPU, Neural Engine)
        
        guard let model = try? TeacherModel(configuration: config) else {
            fatalError("Failed to load the TeacherModel Core ML model.")
        }
        return model
    }()
    
    /// Error types specific to TeacherModelCoreMLPredictor
    enum PredictionError: Error {
        case preprocessingFailed
        case predictionFailed
        case alreadyRunning
        
        var localizedDescription: String {
            switch self {
            case .preprocessingFailed:
                return "Failed to preprocess the input image"
            case .predictionFailed:
                return "Failed to run model prediction"
            case .alreadyRunning:
                return "Prediction is already in progress"
            }
        }
    }
    
    /// Predicts image quality score for the given image
    /// - Parameter image: Input UIImage to assess
    /// - Returns: Tuple containing quality score (0-1) and inference time in seconds
    /// - Throws: PredictionError if any step fails
    func predict(_ image: UIImage) throws -> (score: Float, inferenceTime: Double) {
        // Check if prediction is already running
        if isRunning {
            throw PredictionError.alreadyRunning
        }
        
        isRunning = true
        let startTime = CACurrentMediaTime()
        
        defer {
            isRunning = false
        }
        
        // Convert UIImage to CVPixelBuffer
        guard let pixelBuffer = image.pixelBuffer() else {
            throw PredictionError.preprocessingFailed
        }
        
        // Preprocess image
        guard let normalizedBuffer = pixelBuffer.normalizedForTeacherModel() else {
            throw PredictionError.preprocessingFailed
        }
        
        // Create MLMultiArray from normalized buffer
        guard let inputArray = try? MLMultiArray(shape: [1, 3, 
                                                       NSNumber(value: TeacherModelCoreMLConstants.inputHeight),
                                                       NSNumber(value: TeacherModelCoreMLConstants.inputWidth)],
                                               dataType: .float32) else {
            throw PredictionError.preprocessingFailed
        }
        
        // Copy normalized data to MLMultiArray
        let ptr = UnsafeMutablePointer<Float32>(OpaquePointer(inputArray.dataPointer))
        normalizedBuffer.withUnsafeBufferPointer { buffer in
            ptr.initialize(from: buffer.baseAddress!, count: normalizedBuffer.count)
        }
        
        // Run prediction using the generated model class
        do {
            let output = try model.prediction(input_image: inputArray)
            let score = output.var_2945[0].floatValue
            let inferenceTime = CACurrentMediaTime() - startTime
            return (score: score, inferenceTime: inferenceTime)
        } catch {
            throw PredictionError.predictionFailed
        }
    }
    
    /// Predicts image quality score with a completion handler
    /// - Parameters:
    ///   - image: Input UIImage to assess
    ///   - completion: Completion handler with Result type
    func predictAsync(_ image: UIImage, completion: @escaping (Result<(score: Float, inferenceTime: Double), Error>) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let result = try self.predict(image)
                DispatchQueue.main.async {
                    completion(.success(result))
                }
            } catch {
                DispatchQueue.main.async {
                    completion(.failure(error))
                }
            }
        }
    }
    
    /// Helper method to debug preprocessing
    /// - Parameter image: Input UIImage to preprocess
    /// - Returns: Preprocessed UIImage for visualization, or nil if preprocessing fails
    func debugPreprocessing(_ image: UIImage) -> UIImage? {
        guard let pixelBuffer = image.pixelBuffer() else {
            return nil
        }
        return pixelBuffer.debugPreprocessedImage()
    }
}

// MARK: - Usage Example
extension TeacherModelCoreMLPredictor {
    static func example() {
        let predictor = TeacherModelCoreMLPredictor()
        
        // Synchronous usage
        if let image = UIImage(named: "sample_image") {
            do {
                let (score, time) = try predictor.predict(image)
                print("Quality Score: \(score), Inference Time: \(time)s")
            } catch {
                print("Prediction failed: \(error.localizedDescription)")
            }
        }
        
        // Asynchronous usage
        if let image = UIImage(named: "sample_image") {
            predictor.predictAsync(image) { result in
                switch result {
                case .success(let (score, time)):
                    print("Quality Score: \(score), Inference Time: \(time)s")
                case .failure(let error):
                    print("Prediction failed: \(error.localizedDescription)")
                }
            }
        }
    }
}
