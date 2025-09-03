#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface TorchModule : NSObject

- (nullable instancetype)initWithFileAtPath:(NSString*)filePath
    NS_SWIFT_NAME(init(fileAtPath:))NS_DESIGNATED_INITIALIZER;
- (nullable instancetype)initWithFileAtPath:(NSString*)filePath subFilePath:(NSString*)subFilePath
    NS_SWIFT_NAME(init(fileAtPath:subFilePath:))NS_DESIGNATED_INITIALIZER;
+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;

@end

@interface MUSIQTorchModule : TorchModule
- (float)predictImage:(void*)imageBuffer
                 size:(CGSize)size
           scaledImg1:(void*)scaledImageBuffer1
                size1:(CGSize)size1
           scaledImg2:(void*)scaledImageBuffer2
                size2:(CGSize)size2
NS_SWIFT_NAME(predict(image:size:scaled1:size1:scaled2:size2:));
@end

@interface VisionTorchModule : TorchModule
- (nullable NSArray<NSNumber*>*)predictImage:(void*)imageBuffer NS_SWIFT_NAME(predict(image:));
@end

@interface NLPTorchModule : TorchModule
- (nullable NSArray<NSString*>*)topics;
- (nullable NSArray<NSNumber*>*)predictText:(NSString*)text NS_SWIFT_NAME(predict(text:));
@end

@interface TeacherTorchModule : TorchModule
- (NSArray<NSArray<NSObject*>*>*)predictImage:(void*)imageBuffer
                                        size:(CGSize)size
NS_SWIFT_NAME(predict(image:size:));
@end

NS_ASSUME_NONNULL_END
