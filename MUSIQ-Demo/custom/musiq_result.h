//
//  musiq_result.h
//  MUSIQ-Demo
//
//  Created by Gavin Xiang on 2023/4/18.
//

#ifndef TENSORFLOW_LITE_SUPPORT_C_TASK_PROCESSOR_MUSIQ_RESULT_H_
#define TENSORFLOW_LITE_SUPPORT_C_TASK_PROCESSOR_MUSIQ_RESULT_H_

// Defines C structure for MUSIQ Results and associated helper methods.

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Holds Image MUSIQ results.
// Contains one set of results per image classifier head.
typedef struct TfLiteMUSIQResult {
  // mean opinion score
  int mos;
} TfLiteMUSIQResult;

// Frees up the MUSIQResult Structure.
void TfLiteMUSIQResultDelete(
    TfLiteMUSIQResult* musiq_result);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif /* TENSORFLOW_LITE_SUPPORT_C_TASK_PROCESSOR_MUSIQ_RESULT_H_ */
