//
//  CameraManager.swift
//  Split-It
//
//  Created by Isaac Tennant on 18/07/2025.
//

import Foundation
import AVFoundation

class CameraManager: NSObject {
    private let captureSession = AVCaptureSession()
    
    private var deviceInput: AVCaptureDeviceInput?
    
    private var videoOutput: AVCaptureVideoDataOutput?
    
    private let systemPreferredCamera = AVCaptureDevice.default(for: .video)
    
    private var sessionQueue = DispatchQueue(label: "video.preview.session")
    
    private var addToPreviewStream: ((CGImage) -> Void)?
    
    private var isAuthorized: Bool {
        get async {
            let status = AVCaptureDevice.authorizationStatus(for: .video)
            
            // Determine if the user previously authorized camera access.
            var isAuthorized = status == .authorized
            
            // If the system hasn't determined the user's authorization status,
            // explicitly prompt them for approval.
            if status == .notDetermined {
                isAuthorized = await AVCaptureDevice.requestAccess(for: .video)
            }
            
            return isAuthorized
        }
    }
    
    lazy var previewStream: AsyncStream<CGImage> = {
        AsyncStream { continuation in
            addToPreviewStream = { cgImage in
                continuation.yield(cgImage)
            }
        }
    }()
    
    override init() {
            super.init()
            
            Task {
                await configureSession()
                await startSession()
            }
        }
        
        private func configureSession() async {
            
        }
        
        private func startSession() async {
               
        }

}
