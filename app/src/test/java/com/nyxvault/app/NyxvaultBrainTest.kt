package com.nyxvault.app

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class NyxvaultBrainTest {
    private val brain = NyxvaultBrain()

    @Test
    fun mathToolsArePartOfTheUnifiedBrain() {
        val answer = brain.answer("2 + 2", emptyList())
        assertTrue(answer.contains("4"))
    }

    @Test
    fun verifiedKnowledgeIsRetrievedByTheUnifiedBrain() {
        val knowledge = listOf(
            Example("Что такое Python?", "Python — язык программирования общего назначения.")
        )
        val answer = brain.answer("Что такое Python?", knowledge)
        assertNotNull(answer)
        assertTrue(answer.contains("язык программирования"))
    }

    @Test
    fun unsupportedQuestionsDoNotInventAnAnswerWithoutModelWeights() {
        val answer = brain.answer("Расскажи о квантовой хромодинамике", emptyList())
        assertTrue(answer.contains("пока не нашёл"))
    }

    @Test
    fun exampleDataRemainsSimpleAndShared() {
        val item = Example("question", "answer")
        assertEquals("question", item.question)
        assertEquals("answer", item.answer)
    }
}
