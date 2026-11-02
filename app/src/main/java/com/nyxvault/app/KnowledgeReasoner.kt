package com.nyxvault.app

import kotlin.math.ln
import kotlin.math.sqrt

data class KnowledgeEvidence(val index: Int, val score: Double, val overlap: Set<String>)

/** Evidence-based on-device reasoning over multiple local knowledge entries. */
object KnowledgeReasoner {
    private fun score(query: String, examples: List<Example>): List<KnowledgeEvidence> {
        val terms = IntelligenceEngine.tokens(query)
        if (terms.isEmpty() || examples.isEmpty()) return emptyList()
        val docs = examples.map { IntelligenceEngine.tokens(it.question + " " + it.answer) }
        val n = docs.size.toDouble()
        val df = terms.associateWith { term -> docs.count { term in it } }
        val weights = terms.associateWith { term -> ln(1.0 + (n - (df[term] ?: 0) + 0.5) / ((df[term] ?: 0) + 0.5)) }
        val norm = sqrt(weights.values.sumOf { it * it }).takeIf { it > 0.0 } ?: 1.0
        return docs.mapIndexedNotNull { index, doc ->
            if (doc.isEmpty()) return@mapIndexedNotNull null
            val matched = terms.intersect(doc)
            if (matched.isEmpty()) return@mapIndexedNotNull null
            val weighted = matched.sumOf { term ->
                val tf = doc.count { it == term }.toDouble()
                (weights[term] ?: 0.0) * (tf * 2.2) / (tf + 1.2)
            } / norm
            KnowledgeEvidence(index, weighted + matched.size.toDouble() / terms.size * 0.35, matched)
        }.sortedByDescending { it.score }
    }

    fun answer(question: String, examples: List<Example>): String? {
        if (examples.isEmpty()) return null
        val normalized = IntelligenceEngine.normalize(question)
        val comparison = Regex("чем\\s+(.+?)\\s+отличается\\s+от\\s+(.+?)(?:\\?|$)|разница\\s+между\\s+(.+?)\\s+и\\s+(.+?)(?:\\?|$)").find(normalized)
        if (comparison != null) {
            val left = (comparison.groupValues[1].ifBlank { comparison.groupValues[3] }).trim()
            val right = (comparison.groupValues[2].ifBlank { comparison.groupValues[4] }).trim()
            val leftHit = score(left, examples).firstOrNull { it.score >= 0.18 }
            val rightHit = score(right, examples).firstOrNull { it.score >= 0.18 && it.index != leftHit?.index }
            if (leftHit != null && rightHit != null) return "СРАВНЕНИЕ ПО ЛОКАЛЬНЫМ ЗНАНИЯМ\n\n$left:\n${examples[leftHit.index].answer}\n\n$right:\n${examples[rightHit.index].answer}\n\nГлавное различие можно сформулировать по этим определениям; если нужен точный разбор по критериям, уточни критерий сравнения."
        }
        val hits = score(question, examples)
        val best = hits.firstOrNull() ?: return null
        if (best.score < 0.18) return null
        val primary = examples[best.index].answer
        val wantsDepth = listOf("подробнее", "подробно", "объясни", "расскажи", "почему", "как работает").any { normalized.contains(it) }
        if (!wantsDepth) return primary
        val supporting = hits.drop(1).asSequence()
            .filter { it.score >= 0.18 && it.index != best.index }
            .filter { hit -> hit.overlap.intersect(best.overlap).isNotEmpty() || hit.overlap.size >= 2 }
            .take(2).map { examples[it.index].answer }.distinct().toList()
        return buildString {
            append("ОСНОВНОЙ ОТВЕТ\n").append(primary)
            if (supporting.isNotEmpty()) {
                append("\n\nСВЯЗАННЫЕ СВЕДЕНИЯ\n")
                supporting.forEach { append("• ").append(it).append("\n") }
            }
            append("\nОтвет собран из нескольких подходящих записей локальной базы. Если фактов недостаточно, я не буду выдумывать продолжение.")
        }
    }
}
