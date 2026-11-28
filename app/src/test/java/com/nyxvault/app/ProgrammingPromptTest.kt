package com.nyxvault.app

import org.junit.Assert.assertTrue
import org.junit.Test

class ProgrammingPromptTest {
    @Test
    fun pythonCodeRequestUsesStructuredTrainingFormat() {
        val prompt = ProgrammingPrompt.build("Напиши калькулятор на Python")
        assertTrue(prompt.contains("### Level: 1"))
        assertTrue(prompt.contains("### Topic: functions-basics"))
        assertTrue(prompt.contains("### Task type: write_code"))
        assertTrue(prompt.contains("### Language: python"))
        assertTrue(prompt.endsWith("### Solution\n"))
    }

    @Test
    fun godotGameRequestUsesCompleteProjectFormat() {
        val prompt = ProgrammingPrompt.build("Создай игру на Godot 4")
        assertTrue(prompt.contains("### Level: 8"))
        assertTrue(prompt.contains("### Topic: complete-programs"))
        assertTrue(prompt.contains("### Task type: design"))
        assertTrue(prompt.contains("### Language: gdscript"))
    }


    @Test
    fun greetingUsesTheSameMetadataAsTheTrainingChatExamples() {
        val prompt = ProgrammingPrompt.build("Привет!")
        assertTrue(prompt.contains("### Level: 1"))
        assertTrue(prompt.contains("### Topic: input-output"))
        assertTrue(prompt.contains("### Task type: explain"))
        assertTrue(prompt.contains("### Language: python"))
    }

    @Test
    fun explainingAProgrammingConceptIsNotMisroutedAsCodeGeneration() {
        val prompt = ProgrammingPrompt.build(
            "Объясни простыми словами, что такое переменная в программировании."
        )
        assertTrue(prompt.contains("### Topic: variables-and-types"))
        assertTrue(prompt.contains("### Task type: explain"))
        assertTrue(prompt.contains("### Language: python"))
    }

    @Test
    fun explainingIntAndStrUsesTheVariablesTopic() {
        val prompt = ProgrammingPrompt.build("Чем тип int отличается от str в Python?")
        assertTrue(prompt.contains("### Topic: variables-and-types"))
        assertTrue(prompt.contains("### Task type: explain"))
        assertTrue(prompt.contains("### Language: python"))
    }


    @Test
    fun htmlButtonRequestIsStillRoutedToCodeGeneration() {
        val prompt = ProgrammingPrompt.build(
            "Напиши минимальную HTML-кнопку с текстом Нажми меня."
        )
        assertTrue(prompt.contains("### Topic: concurrency-and-async"))
        assertTrue(prompt.contains("### Task type: write_code"))
        assertTrue(prompt.contains("### Language: javascript"))
    }

    @Test
    fun previousContextIsKeptInsideTheTaskSection() {
        val prompt = ProgrammingPrompt.build("Добавь сохранение", "У нас уже есть игра на Godot.")
        assertTrue(prompt.contains("### Task\n"))
        assertTrue(prompt.contains("У нас уже есть игра на Godot."))
        assertTrue(prompt.contains("Текущая задача").not())
    }
}
