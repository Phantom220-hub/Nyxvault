package com.nyxvault.app

import java.io.Closeable
import java.io.File

/**
 * The one NYXVAULT brain. Retrieval, reasoning, math tools and the optional
 * trained Transformer are coordinated here instead of behaving as separate AIs.
 */
class NyxvaultBrain : Closeable {
    private var transformer: OnnxTransformer? = null
    val hasTransformer: Boolean get() = transformer != null

    fun loadTransformer(modelFile: File, tokenizerFile: File): Boolean {
        val loaded = OnnxTransformer(modelFile, tokenizerFile)
        transformer?.close()
        transformer = loaded
        return true
    }

    fun answer(question: String, examples: List<Example>, conversationContext: String = ""): String {
        LinearEquationSolver.solve(question)?.let { return it }
        MathSolver.solve(question)?.let { return it }
        KnowledgeReasoner.answer(question, examples)?.let { return it }

        val ranked = IntelligenceEngine.rank(question, examples.map { it.question })
        if (ranked != null && ranked.score >= 0.16) {
            val answer = examples[ranked.index].answer
            val framing = IntelligenceEngine.formatKnowledgeAnswer(question, answer, ranked.score)
            val caution = IntelligenceEngine.uncertainty(ranked.score)
            return if (caution.isBlank()) framing else "$framing\n\n$caution"
        }

        val model = transformer
        if (model != null) {
            val prompt = buildString {
                if (conversationContext.isNotBlank()) append(conversationContext.takeLast(1200)).append("\n")
                append("Вопрос: ").append(question).append("\nОтвет:")
            }
            val generated = try { model.generate(prompt).trim() } catch (_: Exception) { "" }
            if (generated.isNotBlank()) {
                return "ЭКСПЕРИМЕНТАЛЬНЫЙ ОТВЕТ NYXVAULT\n$generated\n\n" +
                    "Ответ создан локальной обученной моделью. Проверь код и важные факты перед использованием."
            }
        }

        return "Я пока не нашёл достаточно надёжного ответа в локальной базе.\n\n" +
            "Добавь проверенное объяснение через Teacher Mode. Когда обученная Transformer-модель будет импортирована, " +
            "я смогу использовать её для генерации чернового ответа."
    }

    override fun close() {
        transformer?.close()
        transformer = null
    }
}
