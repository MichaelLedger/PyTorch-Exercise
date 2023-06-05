//
// resnet50_ML_Neural_Network.swift
//
// This file was automatically generated and should not be edited.
//

import CoreML


/// Model Prediction Input Type
@available(macOS 10.15, iOS 13.0, tvOS 13.0, watchOS 6.0, *)
class resnet50_ML_Neural_NetworkInput : MLFeatureProvider {

    /// x as 1 × 3 × 224 × 224 4-dimensional array of floats
    var x: MLMultiArray

    var featureNames: Set<String> {
        get {
            return ["x"]
        }
    }
    
    func featureValue(for featureName: String) -> MLFeatureValue? {
        if (featureName == "x") {
            return MLFeatureValue(multiArray: x)
        }
        return nil
    }
    
    init(x: MLMultiArray) {
        self.x = x
    }

    @available(macOS 12.0, iOS 15.0, tvOS 15.0, watchOS 8.0, *)
    convenience init(x: MLShapedArray<Float>) {
        self.init(x: MLMultiArray(x))
    }

}


/// Model Prediction Output Type
@available(macOS 10.15, iOS 13.0, tvOS 13.0, watchOS 6.0, *)
class resnet50_ML_Neural_NetworkOutput : MLFeatureProvider {

    /// Source provided by CoreML
    private let provider : MLFeatureProvider

    /// var_830 as multidimensional array of floats
    var var_830: MLMultiArray {
        return self.provider.featureValue(for: "var_830")!.multiArrayValue!
    }

    /// var_830 as multidimensional array of floats
    @available(macOS 12.0, iOS 15.0, tvOS 15.0, watchOS 8.0, *)
    var var_830ShapedArray: MLShapedArray<Float> {
        return MLShapedArray<Float>(self.var_830)
    }

    var featureNames: Set<String> {
        return self.provider.featureNames
    }
    
    func featureValue(for featureName: String) -> MLFeatureValue? {
        return self.provider.featureValue(for: featureName)
    }

    init(var_830: MLMultiArray) {
        self.provider = try! MLDictionaryFeatureProvider(dictionary: ["var_830" : MLFeatureValue(multiArray: var_830)])
    }

    init(features: MLFeatureProvider) {
        self.provider = features
    }
}


/// Class for model loading and prediction
@available(macOS 10.15, iOS 13.0, tvOS 13.0, watchOS 6.0, *)
class resnet50_ML_Neural_Network {
    let model: MLModel

    /// URL of model assuming it was installed in the same bundle as this class
    class var urlOfModelInThisBundle : URL {
        let bundle = Bundle(for: self)
        return bundle.url(forResource: "resnet50_ML_Neural_Network", withExtension:"mlmodelc")!
    }

    /**
        Construct resnet50_ML_Neural_Network instance with an existing MLModel object.

        Usually the application does not use this initializer unless it makes a subclass of resnet50_ML_Neural_Network.
        Such application may want to use `MLModel(contentsOfURL:configuration:)` and `resnet50_ML_Neural_Network.urlOfModelInThisBundle` to create a MLModel object to pass-in.

        - parameters:
          - model: MLModel object
    */
    init(model: MLModel) {
        self.model = model
    }

    /**
        Construct resnet50_ML_Neural_Network instance by automatically loading the model from the app's bundle.
    */
    @available(*, deprecated, message: "Use init(configuration:) instead and handle errors appropriately.")
    convenience init() {
        try! self.init(contentsOf: type(of:self).urlOfModelInThisBundle)
    }

    /**
        Construct a model with configuration

        - parameters:
           - configuration: the desired model configuration

        - throws: an NSError object that describes the problem
    */
    convenience init(configuration: MLModelConfiguration) throws {
        try self.init(contentsOf: type(of:self).urlOfModelInThisBundle, configuration: configuration)
    }

    /**
        Construct resnet50_ML_Neural_Network instance with explicit path to mlmodelc file
        - parameters:
           - modelURL: the file url of the model

        - throws: an NSError object that describes the problem
    */
    convenience init(contentsOf modelURL: URL) throws {
        try self.init(model: MLModel(contentsOf: modelURL))
    }

    /**
        Construct a model with URL of the .mlmodelc directory and configuration

        - parameters:
           - modelURL: the file url of the model
           - configuration: the desired model configuration

        - throws: an NSError object that describes the problem
    */
    convenience init(contentsOf modelURL: URL, configuration: MLModelConfiguration) throws {
        try self.init(model: MLModel(contentsOf: modelURL, configuration: configuration))
    }

    /**
        Construct resnet50_ML_Neural_Network instance asynchronously with optional configuration.

        Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

        - parameters:
          - configuration: the desired model configuration
          - handler: the completion handler to be called when the model loading completes successfully or unsuccessfully
    */
    @available(macOS 11.0, iOS 14.0, tvOS 14.0, watchOS 7.0, *)
    class func load(configuration: MLModelConfiguration = MLModelConfiguration(), completionHandler handler: @escaping (Swift.Result<resnet50_ML_Neural_Network, Error>) -> Void) {
        return self.load(contentsOf: self.urlOfModelInThisBundle, configuration: configuration, completionHandler: handler)
    }

    /**
        Construct resnet50_ML_Neural_Network instance asynchronously with optional configuration.

        Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

        - parameters:
          - configuration: the desired model configuration
    */
    @available(macOS 12.0, iOS 15.0, tvOS 15.0, watchOS 8.0, *)
    class func load(configuration: MLModelConfiguration = MLModelConfiguration()) async throws -> resnet50_ML_Neural_Network {
        return try await self.load(contentsOf: self.urlOfModelInThisBundle, configuration: configuration)
    }

    /**
        Construct resnet50_ML_Neural_Network instance asynchronously with URL of the .mlmodelc directory with optional configuration.

        Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

        - parameters:
          - modelURL: the URL to the model
          - configuration: the desired model configuration
          - handler: the completion handler to be called when the model loading completes successfully or unsuccessfully
    */
    @available(macOS 11.0, iOS 14.0, tvOS 14.0, watchOS 7.0, *)
    class func load(contentsOf modelURL: URL, configuration: MLModelConfiguration = MLModelConfiguration(), completionHandler handler: @escaping (Swift.Result<resnet50_ML_Neural_Network, Error>) -> Void) {
        MLModel.load(contentsOf: modelURL, configuration: configuration) { result in
            switch result {
            case .failure(let error):
                handler(.failure(error))
            case .success(let model):
                handler(.success(resnet50_ML_Neural_Network(model: model)))
            }
        }
    }

    /**
        Construct resnet50_ML_Neural_Network instance asynchronously with URL of the .mlmodelc directory with optional configuration.

        Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

        - parameters:
          - modelURL: the URL to the model
          - configuration: the desired model configuration
    */
    @available(macOS 12.0, iOS 15.0, tvOS 15.0, watchOS 8.0, *)
    class func load(contentsOf modelURL: URL, configuration: MLModelConfiguration = MLModelConfiguration()) async throws -> resnet50_ML_Neural_Network {
        let model = try await MLModel.load(contentsOf: modelURL, configuration: configuration)
        return resnet50_ML_Neural_Network(model: model)
    }

    /**
        Make a prediction using the structured interface

        - parameters:
           - input: the input to the prediction as resnet50_ML_Neural_NetworkInput

        - throws: an NSError object that describes the problem

        - returns: the result of the prediction as resnet50_ML_Neural_NetworkOutput
    */
    func prediction(input: resnet50_ML_Neural_NetworkInput) throws -> resnet50_ML_Neural_NetworkOutput {
        return try self.prediction(input: input, options: MLPredictionOptions())
    }

    /**
        Make a prediction using the structured interface

        - parameters:
           - input: the input to the prediction as resnet50_ML_Neural_NetworkInput
           - options: prediction options 

        - throws: an NSError object that describes the problem

        - returns: the result of the prediction as resnet50_ML_Neural_NetworkOutput
    */
    func prediction(input: resnet50_ML_Neural_NetworkInput, options: MLPredictionOptions) throws -> resnet50_ML_Neural_NetworkOutput {
        let outFeatures = try model.prediction(from: input, options:options)
        return resnet50_ML_Neural_NetworkOutput(features: outFeatures)
    }

    /**
        Make a prediction using the convenience interface

        - parameters:
            - x as 1 × 3 × 224 × 224 4-dimensional array of floats

        - throws: an NSError object that describes the problem

        - returns: the result of the prediction as resnet50_ML_Neural_NetworkOutput
    */
    func prediction(x: MLMultiArray) throws -> resnet50_ML_Neural_NetworkOutput {
        let input_ = resnet50_ML_Neural_NetworkInput(x: x)
        return try self.prediction(input: input_)
    }

    /**
        Make a prediction using the convenience interface

        - parameters:
            - x as 1 × 3 × 224 × 224 4-dimensional array of floats

        - throws: an NSError object that describes the problem

        - returns: the result of the prediction as resnet50_ML_Neural_NetworkOutput
    */

    @available(macOS 12.0, iOS 15.0, tvOS 15.0, watchOS 8.0, *)
    func prediction(x: MLShapedArray<Float>) throws -> resnet50_ML_Neural_NetworkOutput {
        let input_ = resnet50_ML_Neural_NetworkInput(x: x)
        return try self.prediction(input: input_)
    }

    /**
        Make a batch prediction using the structured interface

        - parameters:
           - inputs: the inputs to the prediction as [resnet50_ML_Neural_NetworkInput]
           - options: prediction options 

        - throws: an NSError object that describes the problem

        - returns: the result of the prediction as [resnet50_ML_Neural_NetworkOutput]
    */
    func predictions(inputs: [resnet50_ML_Neural_NetworkInput], options: MLPredictionOptions = MLPredictionOptions()) throws -> [resnet50_ML_Neural_NetworkOutput] {
        let batchIn = MLArrayBatchProvider(array: inputs)
        let batchOut = try model.predictions(from: batchIn, options: options)
        var results : [resnet50_ML_Neural_NetworkOutput] = []
        results.reserveCapacity(inputs.count)
        for i in 0..<batchOut.count {
            let outProvider = batchOut.features(at: i)
            let result =  resnet50_ML_Neural_NetworkOutput(features: outProvider)
            results.append(result)
        }
        return results
    }
}
