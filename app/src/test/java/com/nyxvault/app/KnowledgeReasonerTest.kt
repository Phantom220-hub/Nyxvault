package com.nyxvault.app

import org.junit.Assert.*
import org.junit.Test

class KnowledgeReasonerTest {
    private val examples = listOf(
        Example("Что такое список в Python?", "Список list изменяемый: его элементы можно добавлять, удалять и заменять."),
        Example("Что такое кортеж в Python?", "Кортеж tuple — упорядоченная неизменяемая последовательность."),
        Example("Что такое словарь Python?", "Словарь dict хранит пары ключ-значение.")
    )
    @Test fun explainsUsingRelevantKnowledge() {
        val answer = KnowledgeReasoner.answer("Объясни подробнее список Python", examples)
        assertNotNull(answer); assertTrue(answer!!.contains("изменяемый"))
    }
    @Test fun comparesTwoConceptsFromSeparateEvidence() {
        val answer = KnowledgeReasoner.answer("Чем список отличается от кортежа?", examples)
        assertNotNull(answer); assertTrue(answer!!.contains("изменяемый")); assertTrue(answer.contains("неизменяемая"))
    }
    @Test fun refusesUnsupportedQuestion() {
        assertNull(KnowledgeReasoner.answer("Расскажи о квантовой хромодинамике", examples))
    }
}
