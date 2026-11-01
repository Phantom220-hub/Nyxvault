package com.nyxvault.app

import java.util.Locale

object MathSolver {
    fun solve(question: String): String? {
        val cleaned = question.lowercase(Locale.ROOT)
            .replace("сколько будет", "").replace("посчитай", "")
            .replace("вычисли", "").replace("calculate", "").trim()
        if (!cleaned.matches(Regex("[0-9\\s+*/().,\\-]+"))) return null
        val expression = cleaned.replace(',', '.').replace(" ", "")
        if (expression.isBlank() || !expression.any { it in "+-*/" }) return null
        return try {
            val parser = Parser(expression)
            val result = parser.parse()
            if (!result.isFinite()) return "Результат не определён: проверь деление на ноль или слишком большие числа."
            "РЕШЕНИЕ\\nВыражение: $expression\\nПорядок вычисления: сначала скобки, затем умножение и деление, затем сложение и вычитание.\\nОтвет: ${format(result)}"
        } catch (_: Exception) {
            "Не удалось разобрать выражение. Используй числа, +, -, *, / и скобки."
        }
    }

    private fun format(value: Double): String {
        val rounded = String.format(Locale.US, "%.8f", value).trimEnd('0').trimEnd('.')
        return if (rounded.isEmpty() || rounded == "-0") "0" else rounded
    }

    private class Parser(private val source: String) {
        private var index = 0
        fun parse(): Double {
            val value = expression()
            if (index != source.length) error("Unexpected character")
            return value
        }
        private fun expression(): Double {
            var value = term()
            while (index < source.length && (source[index] == '+' || source[index] == '-')) {
                val op = source[index++]
                val right = term()
                value = if (op == '+') value + right else value - right
            }
            return value
        }
        private fun term(): Double {
            var value = factor()
            while (index < source.length && (source[index] == '*' || source[index] == '/')) {
                val op = source[index++]
                val right = factor()
                if (op == '/' && right == 0.0) error("Division by zero")
                value = if (op == '*') value * right else value / right
            }
            return value
        }
        private fun factor(): Double {
            if (index >= source.length) error("Missing number")
            if (source[index] == '+') { index++; return factor() }
            if (source[index] == '-') { index++; return -factor() }
            if (source[index] == '(') {
                index++
                val value = expression()
                if (index >= source.length || source[index] != ')') error("Missing )")
                index++
                return value
            }
            val start = index
            while (index < source.length && (source[index].isDigit() || source[index] == '.')) index++
            if (start == index) error("Number expected")
            return source.substring(start, index).toDouble()
        }
    }
}
