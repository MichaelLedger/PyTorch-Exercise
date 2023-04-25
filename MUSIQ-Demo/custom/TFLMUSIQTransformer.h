//
//  TFLMUSIQTransformer.h
//  MUSIQ-Demo
//
//  Created by Gavin Xiang on 2023/4/18.
//
#import <Foundation/Foundation.h>

#import "TFLMUSIQResult.h"
//#import "TFLBaseOptions.h"
//#import "TFLClassificationOptions.h"
//#import "TFLClassificationResult.h"
//#import "GMLImage.h"

@import TensorFlowLiteTaskVision;

NS_ASSUME_NONNULL_BEGIN

/**
 * Options to configure TFLMUSIQTransformer.
 */
NS_SWIFT_NAME(ImageClassifierOptions)
@interface TFLMUSIQTransformerOptions : NSObject

/**
 * Base options that are used for creation of any type of task.
 * @discussion Please see `TFLBaseOptions` for more details.
 */
@property(nonatomic, copy) TFLBaseOptions *baseOptions;

/**
 * Options that configure the display and filtering of results.
 * @discussion Please see `TFLClassificationOptions` for more details.
 */
@property(nonatomic, copy) TFLClassificationOptions *classificationOptions;

/**
 * Initializes a new `TFLMUSIQTransformerOptions` with the absolute path to the model file
 * stored locally on the device, set to the given the model path.
 *
 * @discussion The external model file, must be a single standalone TFLite file. It could be packed
 * with TFLite Model Metadata[1] and associated files if exist. Fail to provide the necessary
 * metadata and associated files might result in errors. Check the [documentation]
 * (https://www.tensorflow.org/lite/convert/metadata) for each task about the specific requirement.
 *
 * @param modelPath An absolute path to a TensorFlow Lite model file stored locally on the device.
 *
 * @return An instance of `TFLMUSIQTransformerOptions` initialized to the given
 * model path.
 */
- (instancetype)initWithModelPath:(NSString *)modelPath;

@end

/**
 * A TensorFlow Lite Task Image Classifiier.
 */
NS_SWIFT_NAME(ImageClassifier)
@interface TFLMUSIQTransformer : NSObject

/**
 * Creates a new instance of `TFLMUSIQTransformer` from the given `TFLMUSIQTransformerOptions`.
 *
 * @param options The options to use for configuring the `TFLMUSIQTransformer`.
 * @param error An optional error parameter populated when there is an error in initializing
 * the image classifier.
 *
 * @return A new instance of `TFLMUSIQTransformer` with the given options. `nil` if there is an error
 * in initializing the image classifier.
 */
+ (nullable instancetype)transformWithOptions:(TFLMUSIQTransformerOptions *)options
                                        error:(NSError **)error
NS_SWIFT_NAME(classifier(options:));

+ (instancetype)new NS_UNAVAILABLE;

/**
 * Performs classification on the given GMLImage.
 *
 * @discussion This method currently supports classification of only the following types of images:
 * 1. RGB and RGBA images for `GMLImageSourceTypeImage`.
 * 2. kCVPixelFormatType_32BGRA for `GMLImageSourceTypePixelBuffer` and
 *    `GMLImageSourceTypeSampleBuffer`. If you are using `AVCaptureSession` to setup
 *    camera and get the frames for inference, you must request for this format
 *    from AVCaptureVideoDataOutput. Otherwise your classification
 *    results will be wrong.
 *
 * @param image An image to be classified, represented as a `GMLImage`.
 *
 * @return A TFLClassificationResult with one set of results per image classifier head. `nil` if
 * there is an error encountered during classification. Please see `TFLClassificationResult` for
 * more details.
 */
- (nullable TFLMUSIQResult *)transformWithGMLImage:(GMLImage *)image
                                             error:(NSError **)error
NS_SWIFT_NAME(classify(mlImage:));

/**
 * Performs classification on the pixels within the specified region of interest of the given
 * `GMLImage`.
 *
 * @discussion This method currently supports inference on only following type of images:
 * 1. RGB and RGBA images for `GMLImageSourceTypeImage`.
 * 2. kCVPixelFormatType_32BGRA for `GMLImageSourceTypePixelBuffer` and
 *    `GMLImageSourceTypeSampleBuffer`. If you are using `AVCaptureSession` to setup
 *    camera and get the frames for inference, you must request for this format
 *    from AVCaptureVideoDataOutput. Otherwise your classification
 *    results will be wrong.
 *
 * @param image An image to be classified, represented as a `GMLImage`.
 * @param roi A CGRect specifying the region of interest within the given `GMLImage`, on which
 * classification should be performed.
 *
 * @return A TFLClassificationResult with one set of results per image classifier head. `nil` if
 * there is an error encountered during classification.
 */
- (nullable TFLMUSIQResult *)transformWithGMLImage:(GMLImage *)image
                                  regionOfInterest:(CGRect)roi
                                             error:(NSError **)error
NS_SWIFT_NAME(classify(mlImage:regionOfInterest:));

- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END

