//
//  TFLMUSIQTransformer.m
//  MUSIQ-Demo
//
//  Created by Gavin Xiang on 2023/4/18.
//

#import "TFLMUSIQTransformer.h"
#import "TFLCommonUtils.h"
#import "TFLBaseOptions+Helpers.h"
#import "GMLImage+Utils.h"

#include "musiq_transformer.h"

@import TensorFlowLiteTaskVision;

@interface TFLMUSIQTransformer ()
/** TfLiteMUSIQTransformer backed by C API */
@property(nonatomic) TfLiteMUSIQTransformer *musiqTransformer;
@end

@implementation TFLMUSIQTransformerOptions
@synthesize baseOptions;
@synthesize classificationOptions;

- (instancetype)init {
    self = [super init];
    if (self) {
        self.baseOptions = [[TFLBaseOptions alloc] init];
        self.classificationOptions = [[TFLClassificationOptions alloc] init];
    }
    return self;
}

- (instancetype)initWithModelPath:(NSString *)modelPath {
    self = [self init];
    if (self) {
        self.baseOptions.modelFile.filePath = modelPath;
    }
    return self;
}

@end

@implementation TFLMUSIQTransformer
- (void)dealloc {
//    TfLiteMUSIQTransformerDelete(_musiqTransformer);//test
}

- (instancetype)initWithMUSIQTransformer:(TfLiteMUSIQTransformer *)musiqTransformer {
    self = [super init];
    if (self) {
        _musiqTransformer = musiqTransformer;
    }
    return self;
}

+ (nullable instancetype)transformWithOptions:(TFLMUSIQTransformerOptions *)options
                                        error:(NSError **)error {
    if (!options) {
        [TFLCommonUtils createCustomError:error
                                 withCode:TFLSupportErrorCodeInvalidArgumentError
                              description:@"TFLMUSIQTransformerOptions argument cannot be nil."];
        return nil;
    }
    
//    TfLiteMUSIQTransformerOptions cOptions = TfLiteMUSIQTransformerOptionsCreate();
    TfLiteMUSIQTransformerOptions cOptions = {};//test
    
    //test
    //  if (![options.classificationOptions copyToCOptions:&(cOptions.musiq_options)
    //                                               error:error]) {
    //    [options.classificationOptions
    //        deleteAllocatedMemoryOfClassificationOptions:&(cOptions.musiq_options)];
    //    return nil;
    //  }
    
    [options.baseOptions copyToCOptions:&(cOptions.base_options)];
    
    TfLiteSupportError *cCreateClassifierError = NULL;
//    TfLiteMUSIQTransformer *cMUSIQTransformer = TfLiteMUSIQTransformerFromOptions(&cOptions, &cCreateClassifierError);
    TfLiteMUSIQTransformer *cMUSIQTransformer = NULL;//test
    
    //test
    //  [options.classificationOptions
    //      deleteAllocatedMemoryOfClassificationOptions:&(cOptions.classification_options)];
    
    // Populate iOS error if TfliteSupportError is not null and afterwards delete it.
    if (![TFLCommonUtils checkCError:cCreateClassifierError toError:error]) {
        TfLiteSupportErrorDelete(cCreateClassifierError);
    }
    
    // Return nil if classifier evaluates to nil. If an error was generted by the C layer, it has
    // already been populated to an NSError and deleted before returning from the method.
    if (!cMUSIQTransformer) {
        return nil;
    }
    
    return [[TFLMUSIQTransformer alloc] initWithMUSIQTransformer:cMUSIQTransformer];
}

- (nullable TFLMUSIQResult *)transformWithGMLImage:(GMLImage *)image
                                             error:(NSError **)error {
    return [self transformWithGMLImage:image
                      regionOfInterest:CGRectMake(0, 0, image.width, image.height)
                                 error:error];
}

- (nullable TFLMUSIQResult *)transformWithGMLImage:(GMLImage *)image
                                  regionOfInterest:(CGRect)roi
                                             error:(NSError **)error {
    if (!image) {
        [TFLCommonUtils createCustomError:error
                                 withCode:TFLSupportErrorCodeInvalidArgumentError
                              description:@"GMLImage argument cannot be nil."];
        return nil;
    }
    
    TfLiteFrameBuffer *cFrameBuffer = [image cFrameBufferWithError:error];
    
    if (!cFrameBuffer) {
        return nil;
    }
    
    TfLiteBoundingBox boundingBox = {.origin_x = roi.origin.x,
            .origin_y = roi.origin.y,
            .width = roi.size.width,
        .height = roi.size.height};
    
    TfLiteSupportError *transformError = NULL;
//    TfLiteMUSIQResult *cTransformResult = TfLiteMUSIQTransformWithRoi(_musiqTransformer, cFrameBuffer, &boundingBox, &transformError);
    TfLiteMUSIQResult *cTransformResult = NULL;//test
    
    free(cFrameBuffer->buffer);
    cFrameBuffer->buffer = NULL;
    
    free(cFrameBuffer);
    cFrameBuffer = NULL;
    
    // Populate iOS error if C Error is not null and afterwards delete it.
    if (![TFLCommonUtils checkCError:transformError toError:error]) {
        TfLiteSupportErrorDelete(transformError);
    }
    
    // Return nil if C result evaluates to nil. If an error was generted by the C layer, it has
    // already been populated to an NSError and deleted before returning from the method.
    if (!cTransformResult) {
        return nil;
    }
    
    //test
    TFLMUSIQResult *musiqHeadsResults = [TFLMUSIQResult new];
    //  TFLClassificationResult *classificationHeadsResults =
    //      [TFLClassificationResult classificationResultWithCResult:cClassificationResult];
    //  TfLiteClassificationResultDelete(cClassificationResult);
    
    return musiqHeadsResults;
}
@end

