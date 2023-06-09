// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import UIKit
import os
import CoreML

class MUSIQViewController: UIViewController {
    
    /// Image picker for accessing the photo library or camera.
    private var imagePicker = UIImagePickerController()
    
    /// Sample picker
    lazy var slideshow: ImageSlideshow = {
        let slideshow = ImageSlideshow()
        if (slideshow.superview == nil) {
            view.addSubview(slideshow)
        }
        slideshow.backgroundColor = .black
        
        slideshow.slideshowInterval = 0.0//5.0
        slideshow.pageIndicatorPosition = .init(horizontal: .center, vertical: .under)
        slideshow.contentScaleMode = UIViewContentMode.scaleAspectFit

        slideshow.pageIndicator = UIPageControl()

        // optional way to show activity indicator during image load (skipping the line will show no activity indicator)
        slideshow.activityIndicator = DefaultActivityIndicator()
        slideshow.delegate = self

        var localSource: [BundleImageSource] = []
        for image in sampleImageNames {
            localSource.append(BundleImageSource(imageString: image))
        }
        // can be used with other sample sources as `afNetworkingSource`, `alamofireSource` or `sdWebImageSource` or `kingfisherSource`
        slideshow.setImageInputs(localSource)

        let recognizer = UITapGestureRecognizer(target: self, action: #selector(MUSIQViewController.didTap))
        slideshow.addGestureRecognizer(recognizer)
        
        return slideshow
    }()
    
    /// Style transferer instance reponsible for running the TF model. Uses a Int8-based model and
    /// runs inference on the CPU.
//    private var cpuStyleTransferer: StyleTransferer?
//    private var cpuMUSIQTransferer: MUSIQTransferer?
    
    /// Style transferer instance reponsible for running the TF model. Uses a Float16-based model and
    /// runs inference on the GPU.
//    private var gpuStyleTransferer: StyleTransferer?
//    private var gpuMUSIQTransferer: MUSIQTransferer?
    
    private var predictor = MUSIQPredictor()
//    private var coreMLPredictor = MUSIQCoreMLPredictor()
    
    /// Target image to transfer a style onto.
    private var targetImage: UIImage?
    
    /// Style-representative image applied to the input image to create a pastiche.
    private var styleImage: UIImage?
    
    /// Style transfer result.
//    private var styleTransferResult: StyleTransferResult?
//    private var musiqTransferResult: MUSIQTransferResult?
    
    // UI elements
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var photoCameraButton: UIButton!
    @IBOutlet weak var segmentedControl: UISegmentedControl!
    @IBOutlet weak var cropSwitch: UISwitch!
    @IBOutlet weak var useGPUSwitch: UISwitch!
    @IBOutlet weak var inferenceStatusLabel: UILabel!
    @IBOutlet weak var legendLabel: UILabel!
    @IBOutlet weak var styleImageView: UIImageView!
    @IBOutlet weak var runButton: UIButton!
    @IBOutlet weak var pasteImageButton: UIButton!
    @IBOutlet weak var scoreLabel: UILabel!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        imageView.contentMode = .scaleAspectFit
        
        // Setup image picker.
        imagePicker.delegate = self
        imagePicker.sourceType = .photoLibrary
        
        // Set default style image.
//        styleImage = StylePickerDataSource.defaultStyle()
        styleImageView.image = styleImage
        
        // Enable camera option only if current device has camera.
        let isCameraAvailable = UIImagePickerController.isCameraDeviceAvailable(.front)
        || UIImagePickerController.isCameraDeviceAvailable(.rear)
        if isCameraAvailable {
            photoCameraButton.isEnabled = true
        }
        
        // MetalDelegate is not available on iOS Simulator in Xcode versions below 11.
        // If you're not able to run GPU-based inference in iOS simulator, please check
        // your Xcode version.
        useGPUSwitch.isOn = true
        
        // Initialize new style transferer instances.
//        StyleTransferer.newCPUStyleTransferer { result in
//            switch result {
//            case .success(let transferer):
//                self.cpuStyleTransferer = transferer
//            case .error(let wrappedError):
//                print("Failed to initialize: \(wrappedError)")
//            }
//        }
//        StyleTransferer.newGPUStyleTransferer { result in
//            switch result {
//            case .success(let transferer):
//                self.gpuStyleTransferer = transferer
//            case .error(let wrappedError):
//                print("Failed to initialize: \(wrappedError)")
//            }
//        }
//
//        MUSIQTransferer.newCPUMUSIQTransferer { result in
//            switch result {
//            case .success(let transferer):
//                print("[CPU] Success to initialize: \(transferer)")
//                self.cpuMUSIQTransferer = transferer
//            case .error(let wrappedError):
//                print("[CPU] Failed to initialize: \(wrappedError)")
//            }
//        }
//
//        MUSIQTransferer.newGPUMUSIQTransferer { result in
//            switch result {
//            case .success(let transferer):
//                print("[GPU] Success to initialize: \(transferer)")
//                self.gpuMUSIQTransferer = transferer
//            case .error(let wrappedError):
//                print("[GPU] Failed to initialize: \(wrappedError)")
//            }
//        }
    }
    
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        // Observe foregrounding events for pasteboard access.
        addForegroundEventHandler()
        pasteImageButton.isEnabled = imageFromPasteboard() != nil
    }
    
    override func viewDidDisappear(_ animated: Bool) {
        super.viewDidDisappear(animated)
        NotificationCenter.default.removeObserver(self)
    }
    
    @IBAction func onTapPasteImage(_ sender: Any) {
        guard let image = imageFromPasteboard() else { return }
        let actionSheet = imageRoleSelectionAlert(image: image)
        present(actionSheet, animated: true, completion: nil)
    }
    
    @IBAction func onTapSamplePicker(_ sender: Any) {
        openSamplePicker()
    }
    
    @IBAction func onTapRunButton(_ sender: Any) {
        // Make sure that the cached target image is available.
        guard targetImage != nil else {
            self.inferenceStatusLabel.text = "Error: Input image is nil."
            return
        }
        
//        runStyleTransfer(targetImage!)
//        runMUSIQTransfer(targetImage!)
        runMUSIQPredict(targetImage!)
    }
    
    @IBAction func onTapChangeStyleButton(_ sender: Any) {
//        let pickerController = StylePickerViewController.fromStoryboard()
//        pickerController.delegate = self
//        present(pickerController, animated: true, completion: nil)
    }
    
    /// Open camera to allow user taking photo.
    @IBAction func onTapOpenCamera(_ sender: Any) {
        guard
            UIImagePickerController.isCameraDeviceAvailable(.front)
                || UIImagePickerController.isCameraDeviceAvailable(.rear)
        else {
            return
        }
        
        imagePicker.sourceType = .camera
        present(imagePicker, animated: true)
    }
    
    /// Open photo library for user to choose an image from.
    @IBAction func onTapPhotoLibrary(_ sender: Any) {
        imagePicker.sourceType = .photoLibrary
        present(imagePicker, animated: true)
        
//        openPHPicker()
    }
    
    /// Handle tapping on different display mode: Input, Style, Result
    @IBAction func onSegmentChanged(_ sender: Any) {
        switch segmentedControl.selectedSegmentIndex {
        case 0:
            // Mode 0: Show input image
            imageView.image = targetImage
        case 1:
            // Mode 1: Show style image
            imageView.image = styleImage
//        case 2:
            // Mode 2: Show style transfer result.
//            imageView.image = styleTransferResult?.resultImage
        default:
            break
        }
    }
    
    /// Handle changing center crop setting.
    @IBAction func onCropSwitchValueChanged(_ sender: Any) {
        // Make sure that the cached target image is available.
        guard targetImage != nil else {
            self.inferenceStatusLabel.text = "Error: Input image is nil."
            return
        }
        
        // Re-run style transfer upon center-crop setting changed.
//        runStyleTransfer(targetImage!)
//        runMUSIQTransfer(targetImage!)
        runMUSIQPredict(targetImage!)
    }
}

// MARK: - MUSIQ Predict
extension MUSIQViewController {
    func floatValue(data: Data) -> Float32 {
        return Float(bitPattern: UInt32(bigEndian: data.withUnsafeBytes { $0.load(as: UInt32.self) }))
    }
    
    func runMUSIQPredict(_ image: UIImage?) {
        guard let image = image else {
            inferenceStatusLabel.text = "Error: Input image is nil."
            return
        }
        
        // Rotate target image to .up orientation to avoid potential orientation misalignment.
        guard let targetImage = image.transformOrientationToUp() else {
            inferenceStatusLabel.text = "ERROR: Image orientation couldn't be fixed."
            return
        }
        
        self.targetImage = targetImage
//            if styleImage != nil {
//                runStyleTransfer(targetImage)
//                runMUSIQTransfer(targetImage)
//            } else {
//                imageView.image = targetImage
//            }
        imageView.image = targetImage
        
        self.inferenceStatusLabel.text = "Score predict time ..."
        self.runButton.isEnabled = false
        self.scoreLabel.text = "MOS: ..."
//        guard
//            let inputRGBData = image.scaledData(
//                with: Constants.inputImageSize,
//                isQuantized: false
//            )
//        else {
//            print("Failed to convert the input image buffer to RGB data.")
//            return
//        }
        
//        var pixelBuffer: [Float32] = []
//        inputRGBData.withUnsafeBytes { (floatPtr: UnsafePointer<Float32>) in
//            pixelBuffer.append(floatPtr.pointee)
//        }
//        print("pixelBuffer:\n\(pixelBuffer)")
        
        // Pytorch model -- begin
        DispatchQueue.global().async {
            if let results = try? self.predictor.predict(image) {
                DispatchQueue.main.async {
    //                strongSelf.indicator.isHidden = true
    //                strongSelf.bottomView.isHidden = false
    //                strongSelf.benchmarkLabel.isHidden = false
    //                strongSelf.benchmarkLabel.text = String(format: "%.2fms", results.1)
    //                strongSelf.bottomView.update(results: results.0)
                    print("results:\(results)")
//                    self.inferenceStatusLabel.text = String(format: "%.2fms", results.1)
                    self.inferenceStatusLabel.text = "score: \(results.0)\ncost: \(results.1) seconds"
                    self.runButton.isEnabled = true
                    self.scoreLabel.text = "MOS:\(results.0)"
                }
            } else {
                DispatchQueue.main.async {
                    self.inferenceStatusLabel.text = "predict failed!"
                    self.runButton.isEnabled = true
                    self.scoreLabel.text = "MOS:\(0)"
                }
            }
        }
        // Pytorch model -- end
        
        // Core ML - begin
//        DispatchQueue.global().async {
//            self.coreMLPredictor.predict(image: image, completion: { (score, time) in
//                DispatchQueue.main.async {
//                    print("results:\(score),\(time)")
//                    self.inferenceStatusLabel.text = "score: \(score)\ncost: \(time) millisecond"
//                    self.runButton.isEnabled = true
//                    self.scoreLabel.text = "MOS:\(score)"
//                }
//            })
//        }
        // Core ML - end
    }
}

// MARK: - MUSIQ Transfer
extension MUSIQViewController {
//    func runMUSIQTransfer(_ image: UIImage) {
//        clearResults()
        
//        let shouldUseQuantizedFloat16 = useGPUSwitch.isOn
//        let transferer = shouldUseQuantizedFloat16 ? gpuMUSIQTransferer : cpuMUSIQTransferer
//
//        // Make sure that the style transferer is initialized.
//        guard let musiqTransferer = transferer else {
//            inferenceStatusLabel.text = "ERROR: Interpreter is not ready."
//            return
//        }
//
//        guard let targetImage = self.targetImage else {
//            inferenceStatusLabel.text = "ERROR: Select a target image."
//            return
//        }
//
//        // Center-crop the target image if the user has enabled the option.
//        let willCenterCrop = cropSwitch.isOn
//        let image = willCenterCrop ? targetImage.cropCenter() : targetImage
//
//        // Cache the potentially cropped image.
//        self.targetImage = image
//
//        // Show the potentially cropped image on screen.
//        imageView.image = image
//
//        // Make sure that the image is ready before running style transfer.
//        guard image != nil else {
//            inferenceStatusLabel.text = "ERROR: Image could not be cropped."
//            return
//        }
//
//        guard let styleImage = styleImage else {
//            inferenceStatusLabel.text = "ERROR: Select a style image."
//            return
//        }
//
//        // Lock the crop switch and run buttons while style transfer is running.
//        cropSwitch.isEnabled = false
//        runButton.isEnabled = false
//
//        // Run style transfer.
//        musiqTransferer.runMUSIQTransfer(
//            style: styleImage,
//            image: image!,
//            completion: { result in
//                // Show the result on screen
//                switch result {
//                case let .success(musiqTransferResult):
//                    self.musiqTransferResult = musiqTransferResult
//
//                    // Change to show style transfer result
//                    self.segmentedControl.selectedSegmentIndex = 0
//                    self.onSegmentChanged(self)
//
//                    // Show result metadata
////                    self.showInferenceTime(styleTransferResult)
////                    self.showMUSIQResult(musiqTransferResult)
//                case let .error(error):
//                    self.inferenceStatusLabel.text = error.localizedDescription
//                }
//
//                // Regardless of the result, re-enable switching between different display modes
//                self.segmentedControl.isEnabled = true
//                self.cropSwitch.isEnabled = true
//                self.runButton.isEnabled = true
//            })
//    }
}

// MARK: - Style Transfer

extension MUSIQViewController {
    /// Run style transfer on the given image, and show result on screen.
    ///  - Parameter image: The target image for style transfer.
    func runStyleTransfer(_ image: UIImage) {
        clearResults()
        
//        let shouldUseQuantizedFloat16 = useGPUSwitch.isOn
//        let transferer = shouldUseQuantizedFloat16 ? gpuStyleTransferer : cpuStyleTransferer
//
//        // Make sure that the style transferer is initialized.
//        guard let styleTransferer = transferer else {
//            inferenceStatusLabel.text = "ERROR: Interpreter is not ready."
//            return
//        }
//
//        guard let targetImage = self.targetImage else {
//            inferenceStatusLabel.text = "ERROR: Select a target image."
//            return
//        }
//
//        // Center-crop the target image if the user has enabled the option.
//        let willCenterCrop = cropSwitch.isOn
//        let image = willCenterCrop ? targetImage.cropCenter() : targetImage
//
//        // Cache the potentially cropped image.
//        self.targetImage = image
//
//        // Show the potentially cropped image on screen.
//        imageView.image = image
//
//        // Make sure that the image is ready before running style transfer.
//        guard image != nil else {
//            inferenceStatusLabel.text = "ERROR: Image could not be cropped."
//            return
//        }
//
//        guard let styleImage = styleImage else {
//            inferenceStatusLabel.text = "ERROR: Select a style image."
//            return
//        }
//
//        // Lock the crop switch and run buttons while style transfer is running.
//        cropSwitch.isEnabled = false
//        runButton.isEnabled = false
//
//        // Run style transfer.
//        styleTransferer.runStyleTransfer(
//            style: styleImage,
//            image: image!,
//            completion: { result in
//                // Show the result on screen
//                switch result {
//                case let .success(styleTransferResult):
//                    self.styleTransferResult = styleTransferResult
//
//                    // Change to show style transfer result
//                    self.segmentedControl.selectedSegmentIndex = 2
//                    self.onSegmentChanged(self)
//
//                    // Show result metadata
////                    self.showInferenceTime(styleTransferResult)
//                case let .error(error):
//                    self.inferenceStatusLabel.text = error.localizedDescription
//                }
//
//                // Regardless of the result, re-enable switching between different display modes
//                self.segmentedControl.isEnabled = true
//                self.cropSwitch.isEnabled = true
//                self.runButton.isEnabled = true
//            })
    }
    
    /// Clear result from previous run to prepare for new style transfer.
    private func clearResults() {
        inferenceStatusLabel.text = "Running inference with TensorFlow Lite..."
        legendLabel.text = nil
        segmentedControl.isEnabled = false
        segmentedControl.selectedSegmentIndex = 0
    }
    
    /// Show processing time on screen.
//    private func showInferenceTime(_ result: StyleTransferResult) {
//        let timeString = "Preprocessing: \(Int(result.preprocessingTime * 1000))ms.\n"
//        + "Style prediction: \(Int(result.stylePredictTime * 1000))ms.\n"
//        + "Style transfer: \(Int(result.styleTransferTime * 1000))ms.\n"
//        + "Post-processing: \(Int(result.postprocessingTime * 1000))ms.\n"
//
//        inferenceStatusLabel.text = timeString
//    }
    
    /// Show processing time on screen.
//    private func showMUSIQResult(_ result: MUSIQTransferResult) {
//        let timeString = "Preprocessing: \(Int(result.preprocessingTime * 1000))ms.\n"
//        + "Score prediction: \(Int(result.stylePredictTime * 1000))ms.\n"
////        + "Style transfer: \(Int(result.styleTransferTime * 1000))ms.\n"
////        + "Post-processing: \(Int(result.postprocessingTime * 1000))ms.\n"
//        + "mean opinion score: \(result.score)"
//
//        inferenceStatusLabel.text = timeString
//
//        scoreLabel.text = "MOS:\(result.score)"
//    }
}

// MARK: - UIImagePickerControllerDelegate

extension MUSIQViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    
    func imagePickerController(
        _ picker: UIImagePickerController,
        didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]
    ) {
        
        if let pickedImage = info[.originalImage] as? UIImage {
            runMUSIQPredict(pickedImage)
        }
        
        dismiss(animated: true)
    }
}

// MARK: StylePickerViewControllerDelegate

//extension MUSIQViewController: StylePickerViewControllerDelegate {
//
//    func picker(_: StylePickerViewController, didSelectStyle image: UIImage) {
//        styleImage = image
//        styleImageView.image = image
//
//        if let targetImage = targetImage {
////            runStyleTransfer(targetImage)
//            runMUSIQTransfer(targetImage)
//        }
//    }
//
//}

// MARK: Pasteboard images

extension MUSIQViewController {
    
    fileprivate func imageFromPasteboard() -> UIImage? {
        return UIPasteboard.general.images?.first
    }
    
    fileprivate func imageRoleSelectionAlert(image: UIImage) -> UIAlertController {
        let controller = UIAlertController(title: "Paste Image",
                                           message: nil,
                                           preferredStyle: .actionSheet)
        controller.popoverPresentationController?.sourceView = view
        let setInputAction = UIAlertAction(title: "Set input image", style: .default) { _ in
            self.runMUSIQPredict(image)
        }
//        let setStyleAction = UIAlertAction(title: "Set style image", style: .default) { _ in
//            guard let croppedImage = image.cropCenter() else {
//                self.inferenceStatusLabel.text = "ERROR: Unable to crop style image."
//                return
//            }
//
//            self.styleImage = croppedImage
//            self.styleImageView.image = croppedImage
//        }
        let cancelAction = UIAlertAction(title: "Cancel", style: .cancel) { _ in
            controller.dismiss(animated: true, completion: nil)
        }
        controller.addAction(setInputAction)
//        controller.addAction(setStyleAction)
        controller.addAction(cancelAction)
        
        return controller
    }
    
    fileprivate func addForegroundEventHandler() {
        NotificationCenter.default.addObserver(self,
                                               selector: #selector(onForeground(_:)),
                                               name: UIApplication.willEnterForegroundNotification,
                                               object: nil)
    }
    
    @objc fileprivate func onForeground(_ sender: Any) {
        self.pasteImageButton.isEnabled = self.imageFromPasteboard() != nil
    }
    
}

import Photos
import PhotosUI

// MARK: - PHPicker Configurations (PHPickerViewControllerDelegate)
extension MUSIQViewController: PHPickerViewControllerDelegate {
    func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
         picker.dismiss(animated: true, completion: .none)
         results.forEach { result in
               result.itemProvider.loadObject(ofClass: UIImage.self) { reading, error in
               guard let image = reading as? UIImage, error == nil else { return }
               DispatchQueue.main.async {
                   // TODO: - Here you get UIImage
                   self.runMUSIQPredict(image)
               }
//               result.itemProvider.loadFileRepresentation(forTypeIdentifier: "public.image") { [weak self] url, _ in
//                   // TODO: - Here You Get The URL
//               }
          }
       }
  }

   /// call this method for `PHPicker`
   func openPHPicker() {
       var phPickerConfig = PHPickerConfiguration(photoLibrary: .shared())
       phPickerConfig.selectionLimit = 1
       phPickerConfig.filter = PHPickerFilter.any(of: [.images, .livePhotos])
       let phPickerVC = PHPickerViewController(configuration: phPickerConfig)
       phPickerVC.delegate = self
       present(phPickerVC, animated: true)
   }
}

import ImageSlideshow

let sampleImageNames = ["826373",
                       "2017266",
                       "2190310",
                       "2313142",
                       "2484057",
                       "2704811",
                       "3039024",
                       "3615562",
                       "3620726",
                       "3628043"]

// MARK: - Sample Picker
extension MUSIQViewController {
    func openSamplePicker() {
        slideshow.isHidden = false
        slideshow.frame = CGRectMake(0, CGRectGetMaxY(self.view.frame), CGRectGetWidth(self.view.frame), CGRectGetHeight(self.view.frame))
        slideshow.alpha = 0.0
        UIView.animate(withDuration: 0.5, delay: 0.0, options: .curveEaseOut) {
            self.slideshow.frame = CGRectMake(0, 0, CGRectGetWidth(self.view.frame), CGRectGetHeight(self.view.frame))
            self.slideshow.alpha = 1.0
        }
    }
    
    @objc func didTap() {
//        let fullScreenController = slideshow.presentFullScreenController(from: self)
//        // set the activity indicator for full screen controller (skipping the line will show no activity indicator)
//        fullScreenController.slideshow.activityIndicator = DefaultActivityIndicator(style: .white, color: nil)
        
        slideshow.frame = CGRectMake(0, 0, CGRectGetWidth(self.view.frame), CGRectGetHeight(self.view.frame))
        slideshow.alpha = 1.0
        UIView.animate(withDuration: 0.5, delay: 0.0, options: .curveEaseOut) {
            self.slideshow.frame = CGRectMake(0, CGRectGetMaxY(self.view.frame), CGRectGetWidth(self.view.frame), CGRectGetHeight(self.view.frame))
            self.slideshow.alpha = 0.0
        } completion: { _ in
            self.slideshow.isHidden = true
            
            guard self.slideshow.currentPage < sampleImageNames.count else {
                return
            }
            let imageName = sampleImageNames[self.slideshow.currentPage]
            guard let image: UIImage = UIImage(named: imageName) else {
                return
            }
            self.runMUSIQPredict(image)
        }
    }
}

extension MUSIQViewController: ImageSlideshowDelegate {
    func imageSlideshow(_ imageSlideshow: ImageSlideshow, didChangeCurrentPageTo page: Int) {
//        print("current page:", page)
    }
}

// MARK: - Constants
public enum MUSIQConstants {
    // static let inputImageSize = CGSize(width: 224, height: 224)
    
    /*
    32/24: 1024 * 768 = 786432, 786432 * 2 = 1572864
    12/9: 384 * 288 = 110592, 110592 * 2 = 221184
    7/5: 224 * 160 = 35840, 35840 * 2 = 71680
     */
    static let inputImageWidth: Int = 1024 // original
    static let inputImageHeight: Int = 768 // original
    
    static let inputImageWidth1: Int = 384 // multi-scale 1
    static let inputImageHeight1: Int = 288 // multi-scale 1
    
    static let inputImageWidth2: Int = 224 // multi-scale 2
    static let inputImageHeight2: Int = 160 // multi-scale 2
}

