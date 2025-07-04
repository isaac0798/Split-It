//
//  ContentView.swift
//  Split-It
//
//  Created by Isaac Tennant on 04/07/2025.
//

import SwiftUI

struct ContentView: View {
    @State var tapCount = 0
    
    var body: some View {
        Form {
            Section {
                Text("Lets Upload a video!")
            }

            Button("Tap Count: \(tapCount)") {
                self.tapCount += 1
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
