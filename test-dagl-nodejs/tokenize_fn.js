exports.handler = async (input, context) => {
  const text = input.text || '';
  const words = text.toLowerCase().split(/\s+/).filter(w => w.length > 0);
  return {
    tokens: words,
    count: words.length,
  };
};
