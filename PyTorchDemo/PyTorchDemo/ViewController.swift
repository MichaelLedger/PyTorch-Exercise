import UIKit

class ViewController: UIViewController {
    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
    }
    
    private func setupUI() {
        let teacherModelButton = UIButton(type: .system)
        teacherModelButton.translatesAutoresizingMaskIntoConstraints = false
        teacherModelButton.setTitle("Teacher Model Demo", for: .normal)
        teacherModelButton.addTarget(self, action: #selector(openTeacherModelDemo), for: .touchUpInside)
        
        view.addSubview(teacherModelButton)
        
        NSLayoutConstraint.activate([
            teacherModelButton.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            teacherModelButton.centerYAnchor.constraint(equalTo: view.centerYAnchor)
        ])
    }
    
    @objc private func openTeacherModelDemo() {
        let storyboard = UIStoryboard(name: "TeacherModel", bundle: nil)
        if let viewController = storyboard.instantiateInitialViewController() {
            present(viewController, animated: true)
        }
    }
}
