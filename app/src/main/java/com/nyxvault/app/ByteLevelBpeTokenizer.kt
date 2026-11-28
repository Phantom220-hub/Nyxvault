package com.nyxvault.app

import org.json.JSONObject
import java.text.Normalizer
import java.util.regex.Pattern

/**
 * Minimal runtime for the exact GPT-2-style ByteLevel BPE tokenizer produced
 * by training/train_tokenizer.py. The export script writes only vocab/merges.
 */
class ByteLevelBpeTokenizer(json: String) {
    private val vocab = HashMap<String, Int>()
    private val reverseVocab = HashMap<Int, String>()
    private val mergeRanks = HashMap<String, Int>()
    private val unknownId: Int
    private val endId: Int?
    private val byteToUnicode = HashMap<Int, Char>()
    private val unicodeToByte = HashMap<Char, Int>()

    private val splitPattern = Pattern.compile(
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",
        Pattern.UNICODE_CHARACTER_CLASS
    )

    init {
        val root = JSONObject(json)
        val vocabJson = root.getJSONObject("vocab")
        val keys = vocabJson.keys()
        while (keys.hasNext()) {
            val token = keys.next()
            val id = vocabJson.getInt(token)
            vocab[token] = id
            reverseVocab[id] = token
        }
        val merges = root.getJSONArray("merges")
        for (index in 0 until merges.length()) {
            val pair = merges.getString(index)
            val separator = pair.indexOf(' ')
            require(separator > 0 && separator < pair.length - 1) { "Invalid BPE merge: $pair" }
            mergeRanks[pair.substring(0, separator) + "\u0000" + pair.substring(separator + 1)] = index
        }
        unknownId = vocab[root.optString("unk_token", "[UNK]")]
            ?: throw IllegalArgumentException("Tokenizer has no [UNK] token")
        endId = vocab[root.optString("eos_token", "[EOS]")]

        // GPT-2 / Hugging Face ByteLevel byte-to-Unicode mapping.
        val visible = ArrayList<Int>()
        visible.addAll(33..126)
        visible.addAll(161..172)
        visible.addAll(174..255)
        val codePoints = visible.toMutableList()
        var extra = 0
        for (byte in 0..255) {
            if (byte !in visible) {
                visible.add(byte)
                codePoints.add(256 + extra)
                extra++
            }
        }
        for (i in visible.indices) {
            val character = codePoints[i].toChar()
            byteToUnicode[visible[i]] = character
            unicodeToByte[character] = visible[i]
        }
    }

    fun encode(text: String): List<Int> {
        val normalized = Normalizer.normalize(text, Normalizer.Form.NFKC)
        val matcher = splitPattern.matcher(normalized)
        val ids = ArrayList<Int>()
        while (matcher.find()) {
            val piece = matcher.group()
            val encoded = StringBuilder()
            for (byte in piece.toByteArray(Charsets.UTF_8)) {
                val unsigned = byte.toInt() and 0xff
                encoded.append(byteToUnicode[unsigned] ?: return listOf(unknownId))
            }
            val parts = encoded.toString().map { it.toString() }.toMutableList()
            applyBpe(parts)
            for (part in parts) ids.add(vocab[part] ?: unknownId)
        }
        return ids
    }

    private fun applyBpe(parts: MutableList<String>) {
        while (parts.size > 1) {
            var bestIndex = -1
            var bestRank = Int.MAX_VALUE
            for (i in 0 until parts.lastIndex) {
                val rank = mergeRanks[parts[i] + "\u0000" + parts[i + 1]] ?: continue
                if (rank < bestRank) {
                    bestRank = rank
                    bestIndex = i
                }
            }
            if (bestIndex < 0) return
            parts[bestIndex] = parts[bestIndex] + parts[bestIndex + 1]
            parts.removeAt(bestIndex + 1)
        }
    }

    fun decode(ids: List<Int>): String {
        val bytes = ArrayList<Byte>()
        for (id in ids) {
            val token = reverseVocab[id] ?: continue
            if (token.startsWith("[") && token.endsWith("]")) continue
            for (character in token) {
                val value = unicodeToByte[character] ?: continue
                bytes.add(value.toByte())
            }
        }
        return bytes.toByteArray().toString(Charsets.UTF_8)
    }

    fun endTokenId(): Int? = endId
}
