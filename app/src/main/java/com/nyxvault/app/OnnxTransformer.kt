package com.nyxvault.app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.Closeable
import java.io.File
import java.nio.LongBuffer
import kotlin.random.Random

/**
 * Runs the single NYXVAULT Transformer exported by training/export_mobile.py.
 * The model is optional until a real checkpoint has been trained and imported.
 */
class OnnxTransformer(
    modelFile: File,
    tokenizerFile: File,
    private val blockSize: Int = 256
) : Closeable {
    private val environment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val tokenizer: ByteLevelBpeTokenizer

    init {
        require(modelFile.isFile && modelFile.length() > 0L) { "ONNX model file is missing or empty" }
        require(tokenizerFile.isFile && tokenizerFile.length() > 0L) { "Tokenizer file is missing or empty" }
        tokenizer = ByteLevelBpeTokenizer(tokenizerFile.readText(Charsets.UTF_8))
        session = environment.createSession(modelFile.absolutePath, OrtSession.SessionOptions())
    }

    fun generate(prompt: String, maxNewTokens: Int = 96, temperature: Float = 0.8f): String {
        val ids = tokenizer.encode(prompt).takeLast(blockSize).toMutableList()
        if (ids.isEmpty()) return ""
        val generated = ArrayList<Int>()
        val random = Random.Default

        repeat(maxNewTokens.coerceIn(1, 160)) {
            val context = ids.takeLast(blockSize).map(Int::toLong).toLongArray()
            val input = OnnxTensor.createTensor(
                environment,
                LongBuffer.wrap(context),
                longArrayOf(1L, context.size.toLong())
            )
            val nextId = try {
                session.run(mapOf("input_ids" to input)).use { result ->
                    val batch = result[0].value as Array<Array<FloatArray>>
                    val logits = batch[0].last()
                    sample(logits, temperature, random)
                }
            } finally {
                input.close()
            }
            if (nextId == tokenizer.endTokenId()) return tokenizer.decode(generated)
            ids.add(nextId)
            generated.add(nextId)
            if (ids.size > blockSize) ids.removeAt(0)
        }
        return tokenizer.decode(generated)
    }

    private fun sample(logits: FloatArray, temperature: Float, random: Random): Int {
        val topK = 12.coerceAtMost(logits.size)
        val candidates = logits.indices.sortedByDescending { logits[it] }.take(topK)
        val scale = temperature.coerceIn(0.15f, 1.5f)
        val maxLogit = candidates.maxOf { logits[it] / scale }
        val weights = candidates.map { kotlin.math.exp((logits[it] / scale - maxLogit).toDouble()) }
        val total = weights.sum()
        var choice = random.nextDouble() * total
        for (i in candidates.indices) {
            choice -= weights[i]
            if (choice <= 0.0) return candidates[i]
        }
        return candidates.last()
    }

    override fun close() {
        session.close()
    }
}
