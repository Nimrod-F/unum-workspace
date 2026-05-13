exports.handler = async (input, context) => {
  const top = (input.topWords || []).map(t => `${t.word}(${t.freq})`).join(', ');
  return {
    report: `Analysis: ${input.totalWords} words, ${input.uniqueWords} unique. Top: ${top}`,
    stats: input,
  };
};
