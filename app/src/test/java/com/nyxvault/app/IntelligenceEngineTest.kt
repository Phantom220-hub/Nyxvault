package com.nyxvault.app

import org.junit.Assert.*
import org.junit.Test

class IntelligenceEngineTest {
    @Test fun normalizesRussianYoAndCase() {
        assertEquals("еще пример", IntelligenceEngine.normalize("ЕЩЁ, пример!"))
    }

    @Test fun ranksRelevantKnowledgeFirst() {
        val questions = listOf("Что такое Python?", "Как устроена клетка?", "Что такое алгоритм?")
        val result = IntelligenceEngine.rank("Объясни Python", questions)
        assertNotNull(result)
        assertEquals(0, result!!.index)
    }

    @Test fun returnsNullForEmptyIndex() {
        assertNull(IntelligenceEngine.rank("Python", emptyList()))
    }

    @Test fun solvesLinearEquation() {
        val answer = LinearEquationSolver.solve("2x + 4 = 10")
        assertNotNull(answer)
        assertTrue(answer!!.contains("x = 3"))
    }

    @Test fun rejectsUnsupportedEquation() {
        assertNull(LinearEquationSolver.solve("x^2 + 2 = 6"))
    }
}
