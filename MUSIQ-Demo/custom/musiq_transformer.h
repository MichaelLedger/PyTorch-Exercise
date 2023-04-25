//
//  musiq_transformer.h
//  MUSIQ-Demo
//
//  Created by Gavin Xiang on 2023/4/18.
//

#ifndef TENSORFLOW_LITE_SUPPORT_C_TASK_VISION_MUSIQ_TRANSFORMER_H
#define TENSORFLOW_LITE_SUPPORT_C_TASK_VISION_MUSIQ_TRANSFORMER_H

#include <stdint.h>

#include "common.h"
#include "musiq_options.h"
#include "musiq_result.h"
#include "base_options.h"
#include "frame_buffer.h"
#include "bounding_box.h"

@import TensorFlowLiteTaskVision;

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct TfLiteMUSIQTransformer TfLiteMUSIQTransformer;

typedef struct TfLiteMUSIQTransformerOptions {
  TfLiteMUSIQOptions musiq_options;
  TfLiteBaseOptions base_options;
} TfLiteMUSIQTransformerOptions;

// Creates and returns TfLiteMUSIQTransformerOptions initialized with default
// values. Default values are as follows:
// 1. .MUSIQ_options.max_results = -1, which returns all MUSIQ
// categories by default.
// 2. .base_options.compute_settings.tflite_settings.cpu_settings.num_threads =
// -1, which makes the TFLite runtime choose the value.
// 3. .MUSIQ_options.score_threshold = 0
// 4. All pointers like .base_options.model_file.file_path,
// .base_options.MUSIQ_options.display_names_local,
// .MUSIQ_options.label_allowlist.list,
// options.MUSIQ_options.label_denylist.list are NULL.
// 5. All other integer values are initialized to 0.
TfLiteMUSIQTransformerOptions TfLiteMUSIQTransformerOptionsCreate(void);

// Creates TfLiteMUSIQTransformer from options.
// .base_options.model_file.file_path in TfLiteMUSIQTransformerOptions should be
// set to the path of the tflite model you wish to create the
// TfLiteMUSIQTransformer with.
// Create TfLiteMUSIQTransformerOptions using
// TfLiteMUSIQTransformerOptionsCreate(). If need be, you can change the default
// values of options for customizing MUSIQ, If options are not created
// in the aforementioned way, you have to make sure that all members are
// initialized to respective default values and all pointer members are zero
// initialized to avoid any undefined behaviour.
//
// Returns the created image classifier in case of success.
// Returns nullptr on failure which happens commonly due to one of the following
// issues:
// 1. file doesn't exist or is not a well formatted.
// 2. options is nullptr.
// 3. Both options.MUSIQ_options.label_denylist and
// options.MUSIQ_options.label_allowlist are non empty. These
// fields are mutually exclusive.
//
// The caller can check if an error was encountered by testing if the returned
// value of the function is null. If the caller doesn't want the reason for
// failure, they can simply pass a NULL for the address of the error pointer as
// shown below:
//
// TfLiteMUSIQTransformer* classifier = TfLiteMUSIQTransformerFromOptions(options,
// NULL);
//
// If the caller wants to be informed of the reason for failure, they must pass
// the adress of a pointer of type TfLiteSupportError to the `error` param as
// shown below:
//
// TfLiteSupport *error = NULL:
// TfLiteMUSIQTransformer* classifier = TfLiteMUSIQTransformerFromOptions(options,
// &error);
//
// In case of unsuccessful execution, Once the function returns, the error
// pointer will point to a struct containing the error information. If error
// info is passed back to the caller, it is the responsibility of the caller to
// free the error struct by calling the following method defined in common.h:
//
// TfLiteSupportErrorDelete(error)
//
TfLiteMUSIQTransformer* TfLiteMUSIQTransformerFromOptions(
    const TfLiteMUSIQTransformerOptions* options, TfLiteSupportError** error);

// Invokes the encapsulated TFLite model and classifies the frame_buffer.
// Returns a pointer to the created MUSIQ result in case of success or
// NULL in case of failure. The caller must test the return value to identify
// success or failure. If the caller doesn't want the reason for failure, they
// can simply pass a NULL for the address of the error pointer as shown below:
//
// TfLiteMUSIQResult* MUSIQ_result =
// TfLiteMUSIQTransformerClassify(&options, NULL);
//
// If the caller wants to be informed of the reason for failure, they must pass
// the adress of a pointer of type TfLiteSupportError to the `error` param as
// shown below:
//
// TfLiteSupport *error = NULL:
// TfLiteMUSIQTransformer* classifier = TfLiteMUSIQTransformerFromOptions(options,
// &error);
//
// In case of unsuccessful execution, Once the function returns, the error
// pointer will point to a struct containing the error information. If error
// info is passed back to the caller, it is the responsibility of the caller to
// free the error struct by calling the following method defined in common.h:
//
// TfLiteSupportErrorDelete(error)
//
TfLiteMUSIQResult* TfLiteMUSIQTransformerClassify(
    const TfLiteMUSIQTransformer* classifier,
    const TfLiteFrameBuffer* frame_buffer, TfLiteSupportError** error);

// Invokes the encapsulated TFLite model and classifies the region of the
// frame_buffer specified by the bounding box. Same as TfLiteMUSIQTransformer*
// TfLiteMUSIQTransformerFromOptions(
//    const TfLiteMUSIQTransformerOptions* options, TfLiteSupportError** error),
//    except that the
// MUSIQ is performed based on the input region of interest. Cropping
// according to this region of interest is prepended to the pre-processing
// operations.
TfLiteMUSIQResult* TfLiteMUSIQTransformWithRoi(
    const TfLiteMUSIQTransformer* classifier,
    const TfLiteFrameBuffer* frame_buffer, const TfLiteBoundingBox* roi,
    TfLiteSupportError** error);

// Disposes off the image classifier.
void TfLiteMUSIQTransformerDelete(TfLiteMUSIQTransformer* classifier);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif /* TENSORFLOW_LITE_SUPPORT_C_TASK_VISION_MUSIQ_TRANSFORMER_H */
