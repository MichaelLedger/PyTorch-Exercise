import UIKit
import Photos
import PhotosUI

class TeacherModelViewController: UIViewController {
    
    // MARK: - Properties
    
    private var imagePicker = UIImagePickerController()
    private var predictor = TeacherModelPredictor()
    private var targetImage: UIImage?
    
    // MARK: - UI Elements
    
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var photoCameraButton: UIButton!
    @IBOutlet weak var inferenceStatusLabel: UILabel!
    @IBOutlet weak var runButton: UIButton!
    @IBOutlet weak var pasteImageButton: UIButton!
    @IBOutlet weak var scoreLabel: UILabel!
    
    // MARK: - Lifecycle Methods
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        setupUI()
        setupImagePicker()
    }
    
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        addForegroundEventHandler()
        pasteImageButton.isEnabled = imageFromPasteboard() != nil
    }
    
    override func viewDidDisappear(_ animated: Bool) {
        super.viewDidDisappear(animated)
        NotificationCenter.default.removeObserver(self)
    }
    
    // MARK: - Setup Methods
    
    private func setupUI() {
        imageView.contentMode = .scaleAspectFit
        
        // Enable camera button if available
        let isCameraAvailable = UIImagePickerController.isCameraDeviceAvailable(.front)
            || UIImagePickerController.isCameraDeviceAvailable(.rear)
        photoCameraButton.isEnabled = isCameraAvailable
    }
    
    private func setupImagePicker() {
        imagePicker.delegate = self
        imagePicker.sourceType = .photoLibrary
    }
    
    // MARK: - Actions
    
    @IBAction func onTapRunButton(_ sender: Any) {
        guard targetImage != nil else {
            inferenceStatusLabel.text = "Error: Input image is nil."
            return
        }
        
        runTeacherModelPredict(targetImage!)
    }
    
    @IBAction func onTapOpenCamera(_ sender: Any) {
        guard UIImagePickerController.isCameraDeviceAvailable(.front)
                || UIImagePickerController.isCameraDeviceAvailable(.rear)
        else {
            return
        }
        
        imagePicker.sourceType = .camera
        present(imagePicker, animated: true)
    }
    
    @IBAction func onTapPhotoLibrary(_ sender: Any) {
        imagePicker.sourceType = .photoLibrary
        present(imagePicker, animated: true)
    }
    
    @IBAction func onTapPasteImage(_ sender: Any) {
        guard let image = imageFromPasteboard() else { return }
        let actionSheet = imageRoleSelectionAlert(image: image)
        present(actionSheet, animated: true, completion: nil)
    }
    
    // MARK: - Model Prediction
    
    func runTeacherModelPredict(_ image: UIImage?) {
        guard let image = image else {
            inferenceStatusLabel.text = "Error: Input image is nil."
            return
        }
        
        // Rotate target image to .up orientation
        guard let targetImage = image.transformOrientationToUp() else {
            inferenceStatusLabel.text = "ERROR: Image orientation couldn't be fixed."
            return
        }
        
        self.targetImage = targetImage
        imageView.image = targetImage
        
        inferenceStatusLabel.text = "Score prediction in progress..."
        runButton.isEnabled = false
        scoreLabel.text = "Score: ..."
        
        // Run prediction in background
        DispatchQueue.global().async {
            if let results = try? self.predictor.predict(image) {
                DispatchQueue.main.async {
                    self.inferenceStatusLabel.text = String(format: "Score: %.2f\nTime: %.2f seconds", results.0, results.1)
                    self.runButton.isEnabled = true
                    self.scoreLabel.text = String(format: "Score: %.2f", results.0)
                }
            } else {
                DispatchQueue.main.async {
                    self.inferenceStatusLabel.text = "Prediction failed!"
                    self.runButton.isEnabled = true
                    self.scoreLabel.text = "Score: N/A"
                }
            }
        }
    }
}

// MARK: - UIImagePickerControllerDelegate
extension TeacherModelViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    func imagePickerController(
        _ picker: UIImagePickerController,
        didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]
    ) {
        if let pickedImage = info[.originalImage] as? UIImage {
            runTeacherModelPredict(pickedImage)
        }
        
        dismiss(animated: true)
    }
}

// MARK: - PHPicker Configurations
extension TeacherModelViewController: PHPickerViewControllerDelegate {
    func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
        picker.dismiss(animated: true, completion: .none)
        results.forEach { result in
            result.itemProvider.loadObject(ofClass: UIImage.self) { reading, error in
                guard let image = reading as? UIImage, error == nil else { return }
                DispatchQueue.main.async {
                    self.runTeacherModelPredict(image)
                }
            }
        }
    }
    
    func openPHPicker() {
        var phPickerConfig = PHPickerConfiguration(photoLibrary: .shared())
        phPickerConfig.selectionLimit = 1
        phPickerConfig.filter = PHPickerFilter.any(of: [.images, .livePhotos])
        let phPickerVC = PHPickerViewController(configuration: phPickerConfig)
        phPickerVC.delegate = self
        present(phPickerVC, animated: true)
    }
}

// MARK: - Pasteboard Support
extension TeacherModelViewController {
    fileprivate func imageFromPasteboard() -> UIImage? {
        return UIPasteboard.general.images?.first
    }
    
    fileprivate func imageRoleSelectionAlert(image: UIImage) -> UIAlertController {
        let controller = UIAlertController(title: "Paste Image",
                                         message: nil,
                                         preferredStyle: .actionSheet)
        controller.popoverPresentationController?.sourceView = view
        let setInputAction = UIAlertAction(title: "Set input image", style: .default) { _ in
            self.runTeacherModelPredict(image)
        }
        let cancelAction = UIAlertAction(title: "Cancel", style: .cancel) { _ in
            controller.dismiss(animated: true, completion: nil)
        }
        controller.addAction(setInputAction)
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
        pasteImageButton.isEnabled = imageFromPasteboard() != nil
    }
}