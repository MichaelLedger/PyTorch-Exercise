//
//  musiq_options.h
//  MUSIQ-Demo
//
//  Created by Gavin Xiang on 2023/4/18.
//

#ifndef TENSORFLOW_LITE_SUPPORT_C_TASK_PROCESSOR_MUSIQ_OPTIONS_H_
#define TENSORFLOW_LITE_SUPPORT_C_TASK_PROCESSOR_MUSIQ_OPTIONS_H_

#include <stdint.h>

// Defines C Struct for MUSIQ Options Shared by All MUSIQ
// Tasks.

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Model backbone config.
typedef struct TfLiteMUSIQModelConfig {
    int hidden_size;
    char* representation_size;
    char* resnet_emb;
    char* transformer;
} TfLiteMUSIQModelConfig;

// Image preprocessing config.
typedef struct TfLiteMUSIQPPConfig {
    int patch_size;
    int patch_stride;
    int hse_grid_size;
    
    // The longer-side length for the resized variants.
    int longer_side_lengths;
    
    // -1 means using all the patches from the full-size image.
    int max_seq_len_from_original_res;
} TfLiteMUSIQPPConfig;

// Holds settings for any single MUSIQ task.
typedef struct TfLiteMUSIQOptions {
    // Model backbone config.
    struct TfLiteMUSIQModelConfig model_config;
    
    // Number of scores to predict. 10 for AVA and 1 for the other datasets.
    int num_classes;
    
    // image preprocessing config.
    struct TfLiteMUSIQPPConfig pp_config;
    
    // model parameters loaded from checkpoint.
    char* params;
    
    // input image path
    char* image_path;
} TfLiteMUSIQOptions;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_SUPPORT_C_TASK_PROCESSOR_MUSIQ_OPTIONS_H_

