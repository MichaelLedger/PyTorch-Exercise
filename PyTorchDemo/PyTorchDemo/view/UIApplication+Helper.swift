//
//  UIApplication+Helper.swift
//  PyTorchDemo
//
//  Created by Gavin Xiang on 2023/6/3.
//

#if os(iOS)

import Foundation
import UIKit

@available(iOS 13.0, *)
extension UIApplication {
    public func findKeyWindow() -> UIWindow? {
        return connectedScenes
            .filter({$0.activationState == .foregroundActive})
            .compactMap({$0 as? UIWindowScene})
            .first?.windows
            .filter({$0.isKeyWindow}).first
    }
}

#endif

