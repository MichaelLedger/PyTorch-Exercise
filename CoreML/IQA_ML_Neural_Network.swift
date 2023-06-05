//
// IQA_ML_Neural_Network.swift
//
// This file was automatically generated and should not be edited.
//

import CoreML


/// Model Prediction Input Type
@available(macOS 11.0, iOS 14.0, tvOS 14.0, watchOS 7.0, *)
class IQA_ML_Neural_NetworkInput : MLFeatureProvider {

    /// mask_inputs as 1 by 912 matrix of floats
    var mask_inputs: MLMultiArray

    /// feat_dis_org as 1 × 2048 × 24 × 32 4-dimensional array of floats
    var feat_dis_org: MLMultiArray

    /// feat_dis_scale_1 as 1 × 2048 × 9 × 12 4-dimensional array of floats
    var feat_dis_scale_1: MLMultiArray

    /// feat_dis_scale_2 as 1 × 2048 × 5 × 7 4-dimensional array of floats
    var feat_dis_scale_2: MLMultiArray

    var featureNames: Set<String> {
        get {
            return ["mask_inputs", "feat_dis_org", "feat_dis_scale_1", "feat_dis_scale_2"]
        }
    }
    
    func featureValue(for featureName: String) -> MLFeatureValue? {
        if (featureName == "mask_inputs") {
            return MLFeatureValue(multiArray: mask_inputs)
        }
        if (featureName == "feat_dis_org") {
            return MLFeatureValue(multiArray: feat_dis_org)
        }
        if (featureName == "feat_dis_scale_1") {
            return MLFeatureValue(multiArray: feat_dis_scale_1)
        }
        if (featureName == "feat_dis_scale_2") {
            return MLFeatureValue(multiArray: feat_dis_scale_2)
        }
        return nil
    }
    
    init(mask_inputs: MLMultiArray, feat_dis_org: MLMultiArray, feat_dis_scale_1: MLMultiArray, feat_dis_scale_2: MLMultiArray) {
        self.mask_inputs = mask_inputs
        self.feat_dis_org = feat_dis_org
        self.feat_dis_scale_1 = feat_dis_scale_1
        self.feat_dis_scale_2 = feat_dis_scale_2
    }

    @available(macOS 12.0, iOS 15.0, tvOS 15.0, watchOS 8.0, *)
    convenience init(mask_inputs: MLShapedArray<Float>, feat_dis_org: MLShapedArray<Float>, feat_dis_scale_1: MLShapedArray<Float>, feat_dis_scale_2: MLShapedArray<Float>) {
        self.init(mask_inputs: MLMultiArray(mask_inputs), feat_dis_org: MLMultiArray(feat_dis_org), feat_dis_scale_1: MLMultiArray(feat_dis_scale_1), feat_dis_scale_2: MLMultiArray(feat_dis_scale_2))
    }

}


/// Model Prediction Output Type
@available(macOS 11.0, iOS 14.0, tvOS 14.0, watchOS 7.0, *)
class IQA_ML_Neural_NetworkOutput : MLFeatureProvider {

    /// Source provided by CoreML
    private let provider : MLFeatureProvider

    /// var_9627 as multidimensional array of floats
    var var_9627: MLMultiArray {
        return self.provider.featureValue(for: "var_9627")!.multiArrayValue!
    }

    /// var_9627 as multidimensional array of floats
    @available(macOS 12.0, iOS 15.0, tvOS 15.0, watchOS 8.0, *)
    var var_9627ShapedArray: MLShapedArray<Float> {
        return MLShapedArray<Float>(self.var_9627)
    }

    var featureNames: Set<String> {
        return self.provider.featureNames
    }
    
    func featureValue(for featureName: String) -> MLFeatureValue? {
        return self.provider.featureValue(for: featureName)
    }

    init(var_9627: MLMultiArray) {
        self.provider = try! MLDictionaryFeatureProvider(dictionary: ["var_9627" : MLFeatureValue(multiArray: var_9627)])
    }

    init(features: MLFeatureProvider) {
        self.provider = features
    }
}


/// Class for model loading and prediction
@available(macOS 11.0, iOS 14.0, tvOS 14.0, watchOS 7.0, *)
class IQA_ML_Neural_Network {
    let model: MLModel

    /// URL of model assuming it was installed in the same bundle as this class
    class var urlOfModelInThisBundle : URL {
        let bundle = Bundle(for: self)
        return bundle.url(forResource: "IQA_ML_Neural_Network", withExtension:"mlmodelc")!
    }

    /**
        Construct IQA_ML_Neural_Network instance with an existing MLModel object.

        Usually the application does not use this initializer unless it makes a subclass of IQA_ML_Neural_Network.
        Such application may want to use `MLModel(contentsOfURL:configuration:)` and `IQA_ML_Neural_Network.urlOfModelInThisBundle` to create a MLModel object to pass-in.

        - parameters:
          - model: MLModel object
    */
    init(model: MLModel) {
        self.model = model
    }

    /**
        Construct a model with configuration

        - parameters:
           - configuration: the desired model configuration

        - throws: an NSError object that describes the problem
    */
    convenience init(configuration: MLModelConfiguration = MLModelConfiguration()) throws {
        try self.init(contentsOf: type(of:self).urlOfModelInThisBundle, configuration: configuration)
    }

    /**
        Construct IQA_ML_Neural_Network instance with explicit path to mlmodelc file
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
        Construct IQA_ML_Neural_Network instance asynchronously with optional configuration.

        Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

        - parameters:
          - configuration: the desired model configuration
          - handler: the completion handler to be called when the model loading completes successfully or unsuccessfully
    */
    class func load(configuration: MLModelConfiguration = MLModelConfiguration(), completionHandler handler: @escaping (Swift.Result<IQA_ML_Neural_Network, Error>) -> Void) {
        return self.load(contentsOf: self.urlOfModelInThisBundle, configuration: configuration, completionHandler: handler)
    }

    /**
        Construct IQA_ML_Neural_Network instance asynchronously with optional configuration.

        Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

        - parameters:
          - configuration: the desired model configuration
    */
    @available(macOS 12.0, iOS 15.0, tvOS 15.0, watchOS 8.0, *)
    class func load(configuration: MLModelConfiguration = MLModelConfiguration()) async throws -> IQA_ML_Neural_Network {
        return try await self.load(contentsOf: self.urlOfModelInThisBundle, configuration: configuration)
    }

    /**
        Construct IQA_ML_Neural_Network instance asynchronously with URL of the .mlmodelc directory with optional configuration.

        Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

        - parameters:
          - modelURL: the URL to the model
          - configuration: the desired model configuration
          - handler: the completion handler to be called when the model loading completes successfully or unsuccessfully
    */
    class func load(contentsOf modelURL: URL, configuration: MLModelConfiguration = MLModelConfiguration(), completionHandler handler: @escaping (Swift.Result<IQA_ML_Neural_Network, Error>) -> Void) {
        MLModel.load(contentsOf: modelURL, configuration: configuration) { result in
            switch result {
            case .failure(let error):
                handler(.failure(error))
            case .success(let model):
                handler(.success(IQA_ML_Neural_Network(model: model)))
            }
        }
    }

    /**
        Construct IQA_ML_Neural_Network instance asynchronously with URL of the .mlmodelc directory with optional configuration.

        Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

        - parameters:
          - modelURL: the URL to the model
          - configuration: the desired model configuration
    */
    @available(macOS 12.0, iOS 15.0, tvOS 15.0, watchOS 8.0, *)
    class func load(contentsOf modelURL: URL, configuration: MLModelConfiguration = MLModelConfiguration()) async throws -> IQA_ML_Neural_Network {
        let model = try await MLModel.load(contentsOf: modelURL, configuration: configuration)
        return IQA_ML_Neural_Network(model: model)
    }

    /**
        Make a prediction using the structured interface

        - parameters:
           - input: the input to the prediction as IQA_ML_Neural_NetworkInput

        - throws: an NSError object that describes the problem

        - returns: the result of the prediction as IQA_ML_Neural_NetworkOutput
    */
    func prediction(input: IQA_ML_Neural_NetworkInput) throws -> IQA_ML_Neural_NetworkOutput {
        return try self.prediction(input: input, options: MLPredictionOptions())
    }

    /**
        Make a prediction using the structured interface

        - parameters:
           - input: the input to the prediction as IQA_ML_Neural_NetworkInput
           - options: prediction options 

        - throws: an NSError object that describes the problem

        - returns: the result of the prediction as IQA_ML_Neural_NetworkOutput
    */
    func prediction(input: IQA_ML_Neural_NetworkInput, options: MLPredictionOptions) throws -> IQA_ML_Neural_NetworkOutput {
        let outFeatures = try model.prediction(from: input, options:options)
        return IQA_ML_Neural_NetworkOutput(features: outFeatures)
    }

    /**
        Make a prediction using the convenience interface

        - parameters:
            - mask_inputs as 1 by 912 matrix of floats
            - feat_dis_org as 1 × 2048 × 24 × 32 4-dimensional array of floats
            - feat_dis_scale_1 as 1 × 2048 × 9 × 12 4-dimensional array of floats
            - feat_dis_scale_2 as 1 × 2048 × 5 × 7 4-dimensional array of floats

        - throws: an NSError object that describes the problem

        - returns: the result of the prediction as IQA_ML_Neural_NetworkOutput
    */
    func prediction(mask_inputs: MLMultiArray, feat_dis_org: MLMultiArray, feat_dis_scale_1: MLMultiArray, feat_dis_scale_2: MLMultiArray) throws -> IQA_ML_Neural_NetworkOutput {
        let input_ = IQA_ML_Neural_NetworkInput(mask_inputs: mask_inputs, feat_dis_org: feat_dis_org, feat_dis_scale_1: feat_dis_scale_1, feat_dis_scale_2: feat_dis_scale_2)
        return try self.prediction(input: input_)
    }

    /**
        Make a prediction using the convenience interface

        - parameters:
            - mask_inputs as 1 by 912 matrix of floats
            - feat_dis_org as 1 × 2048 × 24 × 32 4-dimensional array of floats
            - feat_dis_scale_1 as 1 × 2048 × 9 × 12 4-dimensional array of floats
            - feat_dis_scale_2 as 1 × 2048 × 5 × 7 4-dimensional array of floats

        - throws: an NSError object that describes the problem

        - returns: the result of the prediction as IQA_ML_Neural_NetworkOutput
    */

    @available(macOS 12.0, iOS 15.0, tvOS 15.0, watchOS 8.0, *)
    func prediction(mask_inputs: MLShapedArray<Float>, feat_dis_org: MLShapedArray<Float>, feat_dis_scale_1: MLShapedArray<Float>, feat_dis_scale_2: MLShapedArray<Float>) throws -> IQA_ML_Neural_NetworkOutput {
        let input_ = IQA_ML_Neural_NetworkInput(mask_inputs: mask_inputs, feat_dis_org: feat_dis_org, feat_dis_scale_1: feat_dis_scale_1, feat_dis_scale_2: feat_dis_scale_2)
        return try self.prediction(input: input_)
    }

    /**
        Make a batch prediction using the structured interface

        - parameters:
           - inputs: the inputs to the prediction as [IQA_ML_Neural_NetworkInput]
           - options: prediction options 

        - throws: an NSError object that describes the problem

        - returns: the result of the prediction as [IQA_ML_Neural_NetworkOutput]
    */
    func predictions(inputs: [IQA_ML_Neural_NetworkInput], options: MLPredictionOptions = MLPredictionOptions()) throws -> [IQA_ML_Neural_NetworkOutput] {
        let batchIn = MLArrayBatchProvider(array: inputs)
        let batchOut = try model.predictions(from: batchIn, options: options)
        var results : [IQA_ML_Neural_NetworkOutput] = []
        results.reserveCapacity(inputs.count)
        for i in 0..<batchOut.count {
            let outProvider = batchOut.features(at: i)
            let result =  IQA_ML_Neural_NetworkOutput(features: outProvider)
            results.append(result)
        }
        return results
    }
}
