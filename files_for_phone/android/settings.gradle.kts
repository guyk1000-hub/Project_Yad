pluginManagement {
    repositories {
        gradlePluginPortal()
        google()
        mavenCentral()
    }
}

dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "ProjectYadPhone"

// Pure Kotlin, no Android dependency: the signal pipeline, testable on a PC.
include(":core")
