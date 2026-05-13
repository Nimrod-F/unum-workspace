exports.handler = async (input, context) => {
  const tokens = input.tokens || [];
  const freq = {};
  for (const word of tokens) {
    freq[word] = (freq[word] || 0) + 1;
  }
  const sorted = Object.entries(freq).sort((a, b) => b[1] - a[1]);
  return {
    totalWords: tokens.length,
    uniqueWords: sorted.length,
    topWords: sorted.slice(0, 5).map(([word, count]) => ({ word, freq: count })),
  };
};
