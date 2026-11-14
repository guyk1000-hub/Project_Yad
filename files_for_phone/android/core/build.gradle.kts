import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    kotlin("jvm")
    kotlin("plugin.serialization")
}

// Java 17 bytecode: what Android Gradle Plugin 8.x expects from library modules.
java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}
kotlin {
    compilerOptions { jvmTarget.set(JvmTarget.JVM_17) }
}

dependencies {
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.8.1")
    testImplementation(kotlin("test"))
}

tasks.test {
    useJUnitPlatform()
    // Single source of truth: the vectors and models produced by files_for_phone/tools.
    val phoneDir = rootProject.projectDir.parentFile
    systemProperty("yad.golden.dir", phoneDir.resolve("tools/golden").absolutePath)
    systemProperty("yad.models.dir", phoneDir.resolve("models").absolutePath)
    testLogging {
        events("passed", "failed", "skipped")
        showStandardStreams = true
        exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
    }
}
