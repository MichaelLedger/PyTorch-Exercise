import UIKit

class TeacherModelViewController: UIViewController {
    // MARK: - Outlets
    @IBOutlet private weak var imageView: UIImageView!
    @IBOutlet private weak var inferenceStatusLabel: UILabel!
    @IBOutlet private weak var scoreLabel: UILabel!
    @IBOutlet private weak var photoCameraButton: UIButton!
    @IBOutlet private weak var runButton: UIButton!
    @IBOutlet private weak var pasteImageButton: UIButton!
    
    // MARK: - Properties
    private lazy var torchPredictor = TeacherModelPredictor()
    private lazy var coreMLPredictor = StudentModelCoreMLPredictor() //TeacherModelCoreMLPredictor()
    
    // MARK: - Lifecycle
    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
    }
    
    // MARK: - UI Setup
    private func setupUI() {
        title = "Teacher Model Comparison"
        inferenceStatusLabel.numberOfLines = 0
        scoreLabel.numberOfLines = 0
    }
    
    // MARK: - Actions
    @IBAction private func onTapOpenCamera(_ sender: UIButton) {
        let picker = UIImagePickerController()
        picker.delegate = self
        picker.sourceType = .camera
        present(picker, animated: true)
    }
    
    @IBAction private func onTapPhotoLibrary(_ sender: UIButton) {
        let picker = UIImagePickerController()
        picker.delegate = self
        picker.sourceType = .photoLibrary
        present(picker, animated: true)
    }
    
    @IBAction private func onTapRunButton(_ sender: UIButton) {
        guard let image = imageView.image else {
            inferenceStatusLabel.text = "Please select an image first"
            return
        }
        processImage(image)
    }
    
    @IBAction private func onTapPasteImage(_ sender: UIButton) {
        if let image = UIPasteboard.general.image {
            imageView.image = image
            processImage(image)
        } else {
            inferenceStatusLabel.text = "No image in clipboard"
        }
    }
    
    // MARK: - Image Processing
    private func processImage(_ image: UIImage) {
        // Update UI for processing state
        inferenceStatusLabel.text = "Processing..."
        scoreLabel.text = "Running models..."
        runButton.isEnabled = false
        
        var torchScore: Float?
        var torchTime: Double?
        var coreMLScore: Float?
        var coreMLTime: Double?
        
        let group = DispatchGroup()
        
        // Run PyTorch prediction
        group.enter()
        DispatchQueue.global(qos: .userInitiated).async {
            if let (scores, time) = try? self.torchPredictor.predict(image) {
                torchScore = scores[0].1
                torchTime = time
            }
            group.leave()
        }
        
        // Run Core ML prediction
        group.enter()
        DispatchQueue.global(qos: .userInitiated).async {
            if let (score, time) = try? self.coreMLPredictor.predict(image) {
                coreMLScore = score
                coreMLTime = time
            }
            group.leave()
        }
        
        // Update UI when both predictions are done
        group.notify(queue: .main) {
            self.runButton.isEnabled = true
            
            var resultText = ""
            var scoreText = ""
            
            if let torchScore = torchScore, let torchTime = torchTime {
                resultText += "PyTorch Score: \(String(format: "%.3f", torchScore))\n"
                resultText += "PyTorch Time: \(String(format: "%.3f", torchTime))s\n\n"
            } else {
                resultText += "PyTorch prediction failed\n\n"
            }
            
            if let coreMLScore = coreMLScore, let coreMLTime = coreMLTime {
                resultText += "Core ML Score: \(String(format: "%.3f", coreMLScore))\n"
                resultText += "Core ML Time: \(String(format: "%.3f", coreMLTime))s"
            } else {
                resultText += "Core ML prediction failed"
            }
            
            // Compare results
            if let torchScore = torchScore, let coreMLScore = coreMLScore,
               let torchTime = torchTime, let coreMLTime = coreMLTime {
                let scoreDiff = abs(torchScore - coreMLScore)
                let timeDiff = torchTime - coreMLTime
                let timeImprovement = (torchTime - coreMLTime) / torchTime * 100
                
                scoreText = "Comparison:\n"
                scoreText += "Score Difference: \(String(format: "%.3f", scoreDiff))\n"
                scoreText += "Time Difference: \(String(format: "%.3f", timeDiff))s\n"
                scoreText += "Speed Improvement: \(String(format: "%.1f", timeImprovement))%"
                
                // Color code the comparison
                if timeImprovement > 0 {
                    self.scoreLabel.textColor = .systemGreen
                } else {
                    self.scoreLabel.textColor = .systemRed
                }
            }
            
            self.inferenceStatusLabel.text = resultText
            self.scoreLabel.text = scoreText
        }
    }
}

// MARK: - UIImagePickerControllerDelegate
extension TeacherModelViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        picker.dismiss(animated: true)
        
        if let image = info[.originalImage] as? UIImage {
            imageView.image = image
            processImage(image)
        }
    }
    
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        picker.dismiss(animated: true)
    }
}
