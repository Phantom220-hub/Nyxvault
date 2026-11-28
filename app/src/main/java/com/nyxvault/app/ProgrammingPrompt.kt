package com.nyxvault.app

import java.util.Locale

/**
 * Formats coding requests exactly like the structured records produced by
 * training/data_schema.py. The app and the training pipeline share one prompt shape.
 */
object ProgrammingPrompt {
    fun build(question: String, conversationContext: String = ""): String {
        val text = question.lowercase(Locale.ROOT)
        val projectRequest = listOf(
            "создай игру", "создать игру", "сделай игру", "сделать игру", "собери игру", "создай настоящую игру",
            "создай проект", "создать проект", "полный проект", "напиши игру",
            "make a game", "create a game", "build a project", "full project"
        ).any { text.contains(it) }
        val explicitCodingIntent = listOf(
            "напиши код", "на python", "на питон", "программа", "программ", "код ",
            "скрипт", "функци", "класс ", "алгоритм", "калькулятор", "исправь ошибку",
            "найди ошибку", "godot", "gdscript", "kotlin", "javascript", "typescript",
            "html", "css", "c++", "rust", "android"
        ).any { text.contains(it) }
        val ordinaryConversation = !projectRequest && !explicitCodingIntent &&
            !text.contains("python") && !text.contains("питон")

        val language = when {
            text.contains("godot") || text.contains("gdscript") -> "gdscript"
            text.contains("kotlin") || text.contains("android") -> "kotlin"
            text.contains("typescript") -> "typescript"
            text.contains("javascript") || text.contains("html") ||
                text.contains("css") || text.contains("браузер") -> "javascript"
            text.contains("c++") -> "cpp"
            text.contains("rust") -> "rust"
            text.contains("python") || text.contains("питон") -> "python"
            else -> if (ordinaryConversation) "ru" else "python"
        }

        val level: Int
        val topic: String
        val taskType: String

        if (projectRequest) {
            level = 8
            topic = if (language == "gdscript" || language == "javascript") {
                "complete-programs"
            } else {
                "multi-file-projects"
            }
            taskType = "design"
        } else {
            if (ordinaryConversation) {
                level = 1
                topic = "general-conversation"
                taskType = if (listOf("что такое", "объясни", "расскажи", "почему", "как работает", "что значит")
                    .any { text.contains(it) }) "explain" else "dialogue"
            } else when (language) {
                "gdscript" -> {
                    level = 7
                    topic = "godot-gdscript"
                    taskType = "write_code"
                }
                "kotlin" -> {
                    level = 7
                    topic = "android-kotlin"
                    taskType = "design"
                }
                "javascript", "typescript" -> {
                    level = 6
                    topic = "concurrency-and-async"
                    taskType = "write_code"
                }
                else -> {
                    level = 1
                    topic = "functions-basics"
                    taskType = "write_code"
                }
            }
        }

        val task = buildString {
            if (conversationContext.isNotBlank()) {
                append("Предыдущий контекст:\n")
                append(conversationContext.takeLast(1000))
                append("\n\n")
            }
            append(question.trim())
        }

        return buildString {
            append("### Level: ").append(level).append("\n")
            append("### Topic: ").append(topic).append("\n")
            append("### Task type: ").append(taskType).append("\n")
            append("### Language: ").append(language).append("\n")
            append("### Task\n").append(task).append("\n")
            append("### Solution\n")
        }
    }
}
