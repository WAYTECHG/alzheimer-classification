// Pure inference functions shared by the app and conversion checks.
export function softmax(scores) {
  const maximum = Math.max(...scores);
  const values = scores.map(value => Math.exp(value - maximum));
  const total = values.reduce((sum, value) => sum + value, 0);
  return values.map(value => value / total);
}
export function clinicalVector(model, input) {
  const values = {Age: Number(input.age), Education_Year: Number(input.education), MMSE: Number(input.mmse),
    [`Race_${input.race}`]: 1, [`Gender_${input.gender}`]: 1,
    [`Imputed_Genotype_${input.imputed}`]: 1, [`APOE4_Status_${input.apoe4}`]: 1,
    [`Ethnicity_${input.ethnicity}`]: 1, [`APOE_Genotype_${input.genotype}`]: 1};
  return model.features.map(name => values[name] ?? 0);
}
export function clinicalProbabilities(model, vector) {
  if (vector.length !== model.features.length || vector.some(value => !Number.isFinite(value))) {
    throw new Error('Please enter valid values for every clinical field.');
  }
  return softmax(model.coefficients.map((weights, i) =>
    weights.reduce((sum, weight, j) => sum + weight * vector[j], model.intercept[i])));
}
export function imageTensor(image) {
  const canvas = document.createElement('canvas'); canvas.width = 208; canvas.height = 176;
  const context = canvas.getContext('2d', {willReadFrequently: true});
  context.fillStyle = 'white'; context.fillRect(0, 0, 208, 176);
  context.imageSmoothingEnabled = true; context.imageSmoothingQuality = 'high';
  context.drawImage(image, 0, 0, 208, 176);
  const rgba = context.getImageData(0, 0, 208, 176).data;
  const rgb = new Float32Array(176 * 208 * 3);
  for (let src = 0, dst = 0; src < rgba.length; src += 4) {
    rgb[dst++] = rgba[src]; rgb[dst++] = rgba[src + 1]; rgb[dst++] = rgba[src + 2];
  }
  return rgb; // NHWC, raw RGB values 0..255; deliberately no /255 normalization.
}
