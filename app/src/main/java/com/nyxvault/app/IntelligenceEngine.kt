package com.nyxvault.app

import java.util.Locale
import kotlin.math.ln
import kotlin.math.sqrt

data class RankedKnowledge(val index: Int, val score: Double, val matchedTerms: Set<String>)

object IntelligenceEngine {
    private val stopWords = setOf(
        "что", "это", "как", "для", "или", "мне", "тебе", "ты", "вы", "я", "мы", "они",
        "the", "is", "are", "what", "how", "and", "for", "with", "это", "дар", "чист"
    )

    private val synonyms = mapOf(
        "объясни" to listOf("объяснение", "расскажи", "поясни", "разъясни"),
        "расскажи" to listOf("объясни", "информация", "сведения"),
        "пример" to listOf("задача", "демонстрация", "иллюстрация"),
        "ошибка" to listOf("баг", "проблема", "исключение"),
        "скорость" to listOf("быстрота", "velocity"),
        "нейросеть" to listOf("нейронная", "модель", "ии", "ai"),
        "память" to listOf("запомни", "сохранить", "memory"),
        "салом" to listOf("привет", "здравствуй"),
        "привет" to listOf("салом", "здравствуйте")
    )

    fun normalize(text: String): String = text.lowercase(Locale.ROOT)
        .replace('ё', 'е')
        .replace(Regex("[^\\p{L}\\p{N}_+#.]+"), " ")
        .trim()
        .replace(Regex("\\s+"), " ")

    fun tokens(text: String): Set<String> {
        val raw = Regex("[\\p{L}\\p{N}_+#.]+").findAll(normalize(text))
            .map { it.value }.filter { it.length > 1 && it !in stopWords }.toSet()
        val expanded = raw.toMutableSet()
        raw.forEach { token ->
            synonyms[token]?.let { expanded.addAll(it) }
            stem(token)?.let { expanded.add(it) }
        }
        return expanded
    }

    private fun stem(word: String): String? {
        val suffixes = listOf(
            "иями", "ями", "ами", "ого", "ему", "ыми", "ими", "ать", "ять", "ение", "ения",
            "остью", "ости", "ов", "ев", "ей", "ам", "ям", "ах", "ях", "ый", "ий", "ая", "ое",
            "ые", "ого", "ему", "ом", "ем", "ую", "юю", "а", "я", "ы", "и", "е", "у", "ю"
        )
        val suffix = suffixes.sortedByDescending { it.length }.firstOrNull { word.length - it.length >= 3 && word.endsWith(it) }
        return suffix?.let { word.removeSuffix(it) }
    }

    fun rank(query: String, questions: List<String>): RankedKnowledge? {
        if (questions.isEmpty()) return null
        val q = tokens(query)
        if (q.isEmpty()) return null
        val docs = questions.map { tokens(it) }
        val n = docs.size.toDouble()
        val df = q.associateWith { term -> docs.count { term in it } }
        val weights = q.associateWith { term -> ln(1.0 + (n - df.getValue(term) + 0.5) / (df.getValue(term) + 0.5)) }
        val qNorm = sqrt(weights.values.sumOf { it * it }).takeIf { it > 0.0 } ?: 1.0
        var best: RankedKnowledge? = null
        docs.forEachIndexed { index, doc ->
            if (doc.isEmpty()) return@forEachIndexed
            val matched = q.intersect(doc)
            val overlap = matched.size.toDouble() / q.size
            val exactPhrase = normalize(query).let { normalized ->
                normalized.length > 3 && normalize(questions[index]).contains(normalized)
            }
            val score = matched.sumOf { term ->
                val tf = doc.count { it == term }.toDouble()
                val idf = weights[term] ?: 0.0
                val bm25 = (tf * 2.2) / (tf + 1.2)
                (idf * bm25) / qNorm
            } + overlap * 0.35 + if (exactPhrase) 0.35 else 0.0
            if (best == null || score > best!!.score) best = RankedKnowledge(index, score, matched)
        }
        return best
    }

    fun formatKnowledgeAnswer(question: String, answer: String, score: Double): String {
        val q = normalize(question)
        val asksExplain = listOf("объясни", "поясни", "расскажи подробнее", "чаро", "фахмон").any { q.contains(it) }
        val asksExample = listOf("пример", "например", "мисол").any { q.contains(it) }
        return when {
            asksExplain -> "РАЗБОР\n$answer\n\nЕсли хочешь глубже понять тему, попроси пример или пошаговое объяснение."
            asksExample -> "ОСНОВА\n$answer\n\nПопроси NYXVAULT показать пример на конкретных числах или ситуации."
            else -> answer
        }
    }

    fun uncertainty(score: Double): String = when {
        score < 0.12 -> "Надёжного совпадения в моей локальной базе нет."
        score < 0.24 -> "Похоже на подходящий материал, но совпадение неуверенное."
        else -> ""
    }
}

object LinearEquationSolver {
    private val pattern = Regex("""^\\s*(-?\\d*)\\s*\\*?\\s*x\\s*([+-]\\s*\\d+(?:[.,]\\d+)?)?\\s*=\\s*(-?\\d+(?:[.,]\\d+)?)\\s*$""", RegexOption.IGNORE_CASE)

    fun solve(input: String): String? {
        val cleaned = input.lowercase(Locale.ROOT)
            .replace("реши уравнение", "").replace("реши", "")
            .replace("solve", "").replace("уравнение", "").trim()
            .replace("х", "x")
        val match = pattern.matchEntire(cleaned) ?: return null
        val aText = match.groupValues[1]
        val a = when (aText) { "", "+" -> 1.0; "-" -> -1.0; else -> aText.toDoubleOrNull() ?: return null }
        val b = match.groupValues[2].replace(" ", "").replace(",", ".").toDoubleOrNull() ?: 0.0
        val c = match.groupValues[3].replace(",", ".").toDoubleOrNull() ?: return null
        if (a == 0.0) return "Это не линейное уравнение с единственным неизвестным: коэффициент при x равен нулю."
        val x = (c - b) / a
        if (!x.isFinite()) return "Не удалось получить конечное значение x."
        val fmt = { v: Double -> String.format(Locale.US, "%.6f", v).trimEnd('0').trimEnd('.') }
        return "РЕШЕНИЕ УРАВНЕНИЯ\nИсходное: $cleaned\n1. Переносим свободное число: ${fmt(a)}x = ${fmt(c - b)}\n2. Делим обе части на ${fmt(a)}.\nОтвет: x = ${fmt(x)}\nПроверка: ${fmt(a)} × ${fmt(x)} + ${fmt(b)} = ${fmt(a * x + b)}"
    }
}
