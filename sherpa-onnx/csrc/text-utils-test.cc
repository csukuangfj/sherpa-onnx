// sherpa-onnx/csrc/text-utils-test.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/text-utils.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace sherpa_onnx {

TEST(ToLowerCase, WideString) {
  std::string text =
      "Hallo! Übeltäter übergibt Ärzten öfters äußerst ätzende Öle 3€";
  auto t = ToLowerCase(text);
  std::cout << text << "\n";
  std::cout << t << "\n";
}

TEST(RemoveInvalidUtf8Sequences, Case1) {
  std::vector<uint8_t> v = {
      0xe4, 0xbb, 0x8a,                                  // 今
      0xe5, 0xa4, 0xa9,                                  // 天
      'i',  's',  ' ',  'M', 'o', 'd', 'a', 'y',  ',',   // is Monday,
      ' ',  'w',  'i',  'e', ' ', 'h', 'e', 'i',  0xc3,  // wie heißen Size
      0x9f, 'e',  'n',  ' ', 'S', 'i', 'e', 0xf0, 0x9d, 0x84, 0x81};

  std::vector<uint8_t> v0 = v;
  v0[1] = 0xc0;  // make the first 3 bytes an invalid utf8 character
  std::string s0{v0.begin(), v0.end()};
  EXPECT_EQ(s0.size(), v0.size());

  auto s = RemoveInvalidUtf8Sequences(s0);  // should remove 今

  v0 = v;
  // v0[23] == 0xc3
  // v0[24] == 0x9f

  v0[23] = 0xc1;

  s0 = {v0.begin(), v0.end()};
  s = RemoveInvalidUtf8Sequences(s0);  // should remove ß

  EXPECT_EQ(s.size() + 2, v.size());

  v0 = v;
  // v0[31] = 0xf0;
  // v0[32] = 0x9d;
  // v0[33] = 0x84;
  // v0[34] = 0x81;
  v0[31] = 0xf5;

  s0 = {v0.begin(), v0.end()};
  s = RemoveInvalidUtf8Sequences(s0);

  EXPECT_EQ(s.size() + 4, v.size());
}

// Tests for sanitizeUtf8
TEST(RemoveInvalidUtf8Sequences, ValidUtf8StringPassesUnchanged) {
  std::string input = "Valid UTF-8 🌍";
  EXPECT_EQ(RemoveInvalidUtf8Sequences(input), input);
}

TEST(RemoveInvalidUtf8Sequences, SingleInvalidByteReplaced) {
  std::string input = "Invalid \xFF UTF-8";
  std::string expected = "Invalid  UTF-8";
  EXPECT_EQ(RemoveInvalidUtf8Sequences(input), expected);
}

TEST(RemoveInvalidUtf8Sequences, TruncatedUtf8SequenceReplaced) {
  std::string input = "Broken \xE2\x82";  // Incomplete UTF-8 sequence
  std::string expected = "Broken ";
  EXPECT_EQ(RemoveInvalidUtf8Sequences(input), expected);
}

TEST(RemoveInvalidUtf8Sequences, MultipleInvalidBytes) {
  std::string input = "Test \xC0\xC0\xF8\xA0";  // Multiple invalid sequences
  std::string expected = "Test ";
  EXPECT_EQ(RemoveInvalidUtf8Sequences(input), expected);
}

TEST(RemoveInvalidUtf8Sequences, BreakingCase_SpaceFollowedByInvalidByte) {
  std::string input = "\x20\xC4";  // Space followed by an invalid byte
  std::string expected = " ";      // 0xC4 removed
  EXPECT_EQ(RemoveInvalidUtf8Sequences(input), expected);
}

TEST(RemoveInvalidUtf8Sequences, ValidUtf8WithEdgeCaseCharacters) {
  std::string input = "Edge 🏆💯";
  EXPECT_EQ(RemoveInvalidUtf8Sequences(input), input);
}

TEST(RemoveInvalidUtf8Sequences, MixedValidAndInvalidBytes) {
  std::string input = "Mix \xE2\x82\xAC \xF0\x9F\x98\x81 \xFF";
  std::string expected = "Mix € 😁 ";  // Invalid bytes removed
  EXPECT_EQ(RemoveInvalidUtf8Sequences(input), expected);
}

TEST(RemoveInvalidUtf8Sequences, SpaceFollowedByInvalidByte) {
  std::string input = "\x20\xC4";  // Space (0x20) followed by invalid (0xC4)
  std::string expected = " ";      // Space remains, 0xC4 is removed
  EXPECT_EQ(RemoveInvalidUtf8Sequences(input), expected);
}

TEST(RemoveInvalidUtf8Sequences, RemoveTruncatedC4) {
  std::string input = "Hello \xc4 world";  // Invalid `0xC4`
  std::string expected = "Hello  world";   // `0xC4` should be removed
  EXPECT_EQ(RemoveInvalidUtf8Sequences(input), expected);
}

TEST(RemoveInvalidUtf8Sequences, SpaceFollowedByInvalidByte_Breaking) {
  std::string input = "\x20\xc4";  // Space followed by invalid `0xc4`
  std::string expected = " ";      // `0xc4` should be removed, space remains
  EXPECT_EQ(RemoveInvalidUtf8Sequences(input), expected);
}

TEST(RemoveInvalidUtf8Sequences, DebugSpaceFollowedByInvalidByte) {
  std::string input = "\x20\xc4";  // Space followed by invalid `0xc4`
  std::string output = RemoveInvalidUtf8Sequences(input);

  std::cout << "Processed string: ";
  for (unsigned char c : output) {
    printf("\\x%02x ", c);
  }
  std::cout << std::endl;

  EXPECT_EQ(output, " ");  // Expect `0xc4` to be removed, leaving only space
}

TEST(SplitByAllPunctuation, English) {
  auto result = SplitByAllPunctuation("Hello, world. How are you?");
  ASSERT_EQ(result.size(), 3u);
  EXPECT_EQ(result[0], "Hello,");
  EXPECT_EQ(result[1], "world.");
  EXPECT_EQ(result[2], "How are you?");
}

TEST(SplitByAllPunctuation, Chinese) {
  auto result = SplitByAllPunctuation("你好，世界。你好吗？");
  ASSERT_EQ(result.size(), 3u);
  EXPECT_EQ(result[0], "你好，");
  EXPECT_EQ(result[1], "世界。");
  EXPECT_EQ(result[2], "你好吗？");
}

TEST(SplitByAllPunctuation, MixedPunctuation) {
  auto result = SplitByAllPunctuation("a;b:c,d");
  ASSERT_EQ(result.size(), 4u);
  EXPECT_EQ(result[0], "a;");
  EXPECT_EQ(result[1], "b:");
  EXPECT_EQ(result[2], "c,");
  EXPECT_EQ(result[3], "d");
}

TEST(SplitByAllPunctuation, NoPunctuation) {
  auto result = SplitByAllPunctuation("hello world");
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0], "hello world");
}

TEST(SplitByAllPunctuation, Empty) {
  auto result = SplitByAllPunctuation("");
  EXPECT_EQ(result.size(), 0u);
}

TEST(CountWords, English) {
  EXPECT_EQ(CountWords("hello world"), 2);
  EXPECT_EQ(CountWords("one two three four five"), 5);
  EXPECT_EQ(CountWords(""), 0);
}

TEST(CountWords, Chinese) {
  // Each CJK character is one word
  EXPECT_EQ(CountWords("你好世界"), 4);
  EXPECT_EQ(CountWords("当夜幕降临"), 5);
}

TEST(CountWords, Mixed) {
  // Each CJK char = 1 word, each English word = 1 word
  // "hello"(1) "你"(2) "好"(3) "world"(4) = 4
  EXPECT_EQ(CountWords("hello 你好 world"), 4);
}

TEST(CountWords, Punctuation) {
  // Sentence-ending punctuation counts as a word
  EXPECT_EQ(CountWords("hello."), 2);
  // Commas/semicolons are word boundaries, not words
  EXPECT_EQ(CountWords("a,b,c"), 3);
}

TEST(MergeShortSentencesByWords, Basic) {
  std::vector<std::string> sentences = {"hello", "world"};
  auto result = MergeShortSentencesByWords(sentences, 5);
  // "hello" has 1 word < 5, merged with "world" → "helloworld"
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0], "helloworld");
}

TEST(MergeShortSentencesByWords, LongEnough) {
  std::vector<std::string> sentences = {"one two three four five",
                                        "six seven eight nine ten"};
  auto result = MergeShortSentencesByWords(sentences, 5);
  // Both have >= 5 words, not merged
  ASSERT_EQ(result.size(), 2u);
  EXPECT_EQ(result[0], "one two three four five");
  EXPECT_EQ(result[1], "six seven eight nine ten");
}

TEST(MergeShortSentencesByWords, PunctuationOnlyMerged) {
  std::vector<std::string> sentences = {"hello world", "."};
  auto result = MergeShortSentencesByWords(sentences, 5);
  // "." is all punctuation, merged into previous
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0], "hello world.");
}

TEST(SplitLongSentenceByWords, Short) {
  auto result = SplitLongSentenceByWords("hello world", 20);
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0], "hello world");
}

TEST(SplitLongSentenceByWords, SplitAtSpace) {
  // "a b c d e" has 5 words, max=3 → split at space boundary
  auto result = SplitLongSentenceByWords("a b c d e", 3);
  ASSERT_EQ(result.size(), 2u);
  // The space at the split boundary is consumed
  EXPECT_EQ(result[0], "a b c");
  EXPECT_EQ(result[1], "d e");
}

TEST(SplitLongSentenceByWords, Empty) {
  auto result = SplitLongSentenceByWords("", 20);
  EXPECT_EQ(result.size(), 0u);
}

}  // namespace sherpa_onnx
