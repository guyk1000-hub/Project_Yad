plugins {
    id("com.android.application")
    kotlin("android")
    kotlin("plugin.compose")
}

android {
    namespace = "yad.app"
    compileSdk = 35

    defaultConfig {
        applicationId = "yad.app"
        minSdk = 33          // Galaxy S23 FE launched with Android 13
        targetSdk = 35
        versionCode = 1
        versionName = "0.3.0-m3"
        ndk { abiFilters += "arm64-v8a" }   // the S23 FE; keeps the APK small
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }

    buildFeatures { compose = true }

    // Ship the exported models straight from files_for_phone/models (single source of truth).
    sourceSets["main"].assets.srcDir(rootProject.projectDir.resolve("../models"))
    androidResources {
        noCompress += "tflite"   // TFLite memory-maps the model, which needs it stored uncompressed
        ignoreAssetsPattern = "!.gitignore:!*.md"
    }
}

/** Bundles a few golden vectors as assets/golden/ so the app can check TFLite on the phone itself. */
abstract class CopyGoldenAssets : DefaultTask() {
    @get:InputDirectory abstract val goldenDir: DirectoryProperty
    @get:OutputDirectory abstract val outputDir: DirectoryProperty

    @TaskAction
    fun copy() {
        val out = outputDir.get().asFile.resolve("golden")
        out.deleteRecursively()
        out.mkdirs()
        for (name in listOf("cnn_features.json", "head_tomer.json")) {
            goldenDir.get().asFile.resolve(name).copyTo(out.resolve(name), overwrite = true)
        }
    }
}

val copyGoldenAssets = tasks.register<CopyGoldenAssets>("copyGoldenAssets") {
    goldenDir.set(rootProject.layout.projectDirectory.dir("../tools/golden"))
}

androidComponents {
    onVariants { variant ->
        variant.sources.assets?.addGeneratedSourceDirectory(copyGoldenAssets, CopyGoldenAssets::outputDir)
    }
}

dependencies {
    implementation(project(":core"))

    implementation("org.tensorflow:tensorflow-lite:2.16.1")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.8.1")

    implementation(platform("androidx.compose:compose-bom:2024.12.01"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.7")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.7")
    debugImplementation("androidx.compose.ui:ui-tooling")
}
