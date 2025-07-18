//
//  MainView.swift
//  Split-It
//
//  Created by Isaac Tennant on 18/07/2025.
//

import SwiftUI

struct MainView: View {
    var body: some View {
        TabView {
            AttemptView()
                .tabItem {
                    Label("Attempts", systemImage: "star")
                }
            LeaderboardView()
                .tabItem {
                    Label("LeaderBoard", systemImage: "moon.stars")
                }
            NewAttemptView()
                .tabItem {
                    Label("New Attempt", systemImage: "wind.snow")
                }
            ProfileView()
                .tabItem {
                    Label("Profile", systemImage: "person")
                }
        
        }
        .navigationTitle("Split It!")
    }
}

struct MainView_Previews: PreviewProvider {
    static var previews: some View {
        NavigationView {
            MainView()
        }
    }
}
