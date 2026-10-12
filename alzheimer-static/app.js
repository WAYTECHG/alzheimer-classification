import {clinicalVector, clinicalProbabilities, imageTensor} from './models.js';
const $ = selector => document.querySelector(selector);
const scriptCache = new Map();
function loadScript(path) {
  if (!scriptCache.has(path)) scriptCache.set(path, new Promise((resolve, reject) => {
    const script = document.createElement('script'); script.src = path;
    script.onload = resolve; script.onerror = () => {script.remove(); scriptCache.delete(path); reject(new Error(`Unable to load ${path}. Check the connection and retry.`));};
    document.head.append(script);
  }));
  return scriptCache.get(path);
}
async function readJSON(path) {
  const response = await fetch(path); if (!response.ok) throw new Error(`Unable to load ${path} (${response.status}).`);
  return response.json();
}
const validPages = ['home', 'chatbot', 'clinical', 'mri', 'about'];
function navigate(page) {
  page = validPages.includes(page) ? page : 'home';
  document.querySelectorAll('.page').forEach(section => {section.hidden = section.id !== page;});
  $('#page-picker').value = page;
  if (location.hash !== `#${page}`) history.replaceState(null, '', `#${page}`);
  document.body.classList.remove('nav-open'); $('#menu-toggle').setAttribute('aria-expanded', 'false');
  if (page === 'home') animate('home-animation');
  if (page === 'about') animate('about-animation');
  $('#main').focus({preventScroll:true}); window.scrollTo(0, 0);
}
$('#page-picker').addEventListener('change', event => navigate(event.target.value));
window.addEventListener('hashchange', () => navigate(location.hash.slice(1)));
$('#menu-toggle').addEventListener('click', () => {
  const open = document.body.classList.toggle('nav-open'); $('#menu-toggle').setAttribute('aria-expanded', String(open));
});
document.querySelectorAll('[role=tab]').forEach((button, index, buttons) => {
  function activate() {
    buttons.forEach(tab => {
      const selected = tab === button; tab.classList.toggle('active', selected);
      tab.setAttribute('aria-selected', String(selected)); tab.tabIndex = selected ? 0 : -1;
      $(`#${tab.getAttribute('aria-controls')}`).hidden = !selected;
    });
  }
  button.tabIndex = index ? -1 : 0;
  button.addEventListener('click', activate);
  button.addEventListener('keydown', event => {
    if (['ArrowRight', 'ArrowLeft', 'Home', 'End'].includes(event.key)) {
      event.preventDefault(); const next = event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1 : (index + (event.key === 'ArrowRight' ? 1 : buttons.length - 1)) % buttons.length;
      buttons[next].click(); buttons[next].focus();
    }
  });
});
const animations = new Map();
async function animate(id) {
  if (animations.has(id)) return;
  const container = document.getElementById(id);
  if (!window.lottie) return;
  animations.set(id, true);
  try {
    const animationData = await readJSON('assets/animation.json'); container.replaceChildren();
    const animation = window.lottie.loadAnimation({container, renderer:'svg', loop:true, autoplay:!matchMedia('(prefers-reduced-motion: reduce)').matches, animationData});
    animations.set(id, animation);
  } catch {animations.delete(id);} // The local SVG remains the network-error fallback.
}
const clinicalLabels = {AD:"Alzheimer’s Disease", CN:'Cognitively Normal', LMCI:'Late Mild Cognitive Impairment'};
let clinicalModel;
$('#mmse').addEventListener('input', event => {$('#mmse-value').value = event.target.value;});
function result(container, title, classes, probabilities, note) {
  const index = probabilities.indexOf(Math.max(...probabilities));
  const heading = document.createElement('h2'); heading.textContent = title;
  const label = document.createElement('h3'); label.textContent = classes[index];
  const caption = document.createElement('p'); caption.className = 'muted'; caption.textContent = note;
  const table = document.createElement('table'); table.setAttribute('aria-label', 'Model scores');
  classes.forEach((name, i) => {const row = table.insertRow(); row.insertCell().textContent = name; row.insertCell().textContent = `${(probabilities[i] * 100).toFixed(2)}%`;});
  container.replaceChildren(heading, label, table, caption); container.hidden = false;
}
$('#clinical-form').addEventListener('submit', async event => {
  event.preventDefault(); const button = $('#clinical-predict'); button.disabled = true;
  try {
    clinicalModel ||= await readJSON('models/clinical.json');
    const input = Object.fromEntries(new FormData(event.target));
    const scores = clinicalProbabilities(clinicalModel, clinicalVector(clinicalModel, input));
    result($('#clinical-result'), 'Predicted Clinical Condition:', clinicalModel.classes.map(code => `${clinicalLabels[code]} (${code})`), scores,
      'Saved LogisticRegressionCV model scores. These values are not calibrated diagnostic confidence. A model class prediction is not a diagnosis.');
  } catch (error) {$('#clinical-result').hidden = false; $('#clinical-result').textContent = error.message;}
  finally {button.disabled = false;}
});
let sessionPromise, imageURL, imageReady = false, busy = false;
const MRI_CLASSES = ['Mild Dementia', 'Moderate Dementia', 'Non Dementia', 'Very Mild Dementia'];
async function mriSession() {
  if (!sessionPromise) sessionPromise = (async () => {
    await loadScript('vendor/ort.min.js');
    ort.env.wasm.wasmPaths = new URL('vendor/', location.href).href;
    ort.env.wasm.numThreads = 1; ort.env.wasm.proxy = false;
    return ort.InferenceSession.create('models/mri.onnx', {executionProviders:['wasm'], graphOptimizationLevel:'all'});
  })().catch(error => {sessionPromise = null; throw error;});
  return sessionPromise;
}
async function runMRI() {
  if (!imageReady || busy) return;
  busy = true; $('#mri-predict').disabled = true; $('#mri-file').disabled = true; $('#mri-result').hidden = true;
  try {
    $('#mri-status').textContent = 'Loading the MRI model… the first download may take a moment.';
    const session = await mriSession(); $('#mri-status').textContent = 'Processing the MRI image in your browser…';
    const tensor = new ort.Tensor('float32', imageTensor($('#mri-image')), [1, 176, 208, 3]);
    const outputs = await session.run({[session.inputNames[0]]:tensor});
    const scores = Array.from(outputs[session.outputNames[0]].data);
    if (scores.length !== 4 || scores.some(value => !Number.isFinite(value))) throw new Error('The MRI model returned an invalid output.');
    result($('#mri-result'), 'Predicted Clinical Condition:', MRI_CLASSES, scores,
      'Custom CNN output scores, not calibrated diagnostic confidence. The presentation accuracy has not been reproduced with this checkpoint.');
    $('#mri-status').textContent = 'Prediction complete. Your image was processed locally.';
  } catch (error) {$('#mri-status').textContent = `Unable to predict: ${error.message}. Use a modern browser and click Predict MRI to retry.`;}
  finally {busy = false; $('#mri-predict').disabled = false; $('#mri-file').disabled = false;}
}
async function chooseMRI(file) {
  if (busy || !file) return;
  imageReady = false; $('#mri-preview').hidden = true; $('#mri-result').hidden = true;
  if (!['image/jpeg','image/png'].includes(file.type) || file.size > 10 * 1024 * 1024) {
    $('#mri-status').textContent = 'Choose a JPG or PNG image smaller than 10 MB.'; return;
  }
  if (imageURL) URL.revokeObjectURL(imageURL);
  imageURL = URL.createObjectURL(file); const image = $('#mri-image'); image.src = imageURL;
  try {
    await image.decode();
    if (image.width * image.height > 25_000_000) throw new Error('Image dimensions are too large. Please use an image under 25 megapixels.');
    imageReady = true; $('#mri-preview').hidden = false; await runMRI();
  } catch (error) {$('#mri-status').textContent = `Unable to read this image: ${error.message}`;}
}
$('#mri-file').addEventListener('change', event => chooseMRI(event.target.files[0]));
$('#mri-predict').addEventListener('click', runMRI);
['dragenter','dragover'].forEach(name => $('#drop-zone').addEventListener(name, event => {event.preventDefault(); $('#drop-zone').classList.add('dragover');}));
['dragleave','drop'].forEach(name => $('#drop-zone').addEventListener(name, event => {event.preventDefault(); $('#drop-zone').classList.remove('dragover');}));
$('#drop-zone').addEventListener('drop', event => chooseMRI(event.dataTransfer.files[0]));

// Fixed educational answers. No language model requests or API keys.
const FAQ = [
  {q:'What are the early signs of Alzheimer’s disease?', keys:['early','sign','symptom','memory'], a:'Common signs include memory loss that interferes with daily life and difficulties with familiar tasks, language, or planning. Many conditions can cause these symptoms. Seek a qualified medical assessment for personal concerns.'},
  {q:'How can Alzheimer’s disease be diagnosed?', keys:['diagnos','detect'], a:'Clinicians assess medical history, cognitive abilities, and other possible causes of symptoms. Tests and brain imaging may contribute to an assessment. This project’s predictions cannot establish a diagnosis.'},
  {q:'What are the treatment options for Alzheimer’s?', keys:['treat','medicine','drug'], a:'Treatment decisions require a clinician and depend on an individual’s condition. Care can include medicines, symptom management, and practical support. This offline FAQ does not provide treatment recommendations.'},
  {q:'Can lifestyle changes help prevent Alzheimer’s?', keys:['lifestyle','prevent'], a:'No lifestyle change guarantees prevention. Ask a clinician about healthy activity and managing personal health risks. This FAQ does not estimate your individual risk.'},
  {q:'What is the difference between Alzheimer’s and dementia?', keys:['difference','dementia'], a:'Dementia describes a set of symptoms that affect memory, thinking, and daily functioning. Alzheimer’s disease is one cause of dementia.'},
  {q:'How does Alzheimer’s disease progress over time?', keys:['progress','stage'], a:'Alzheimer’s is a progressive disease, but its course varies between individuals. A care team can help assess changes and plan appropriate support.'},
  {q:'What support is available for caregivers?', keys:['support','caregiver'], a:'Caregivers may find help through local health services, community organizations, support groups, and respite care. Availability depends on location. Ask a local care team about suitable services.'},
  {q:'Are there any new research findings about Alzheimer’s?', keys:['research','new','latest'], a:'This is a fixed offline FAQ and does not contain live research updates. Use current information from the National Institute on Aging or consult a qualified care team.'},
  {q:'What are the risk factors for developing Alzheimer’s?', keys:['risk','factor'], a:'Age, family history, and genetic factors can influence risk. Risk factors do not establish whether someone has Alzheimer’s disease. This educational app cannot estimate a clinically validated personal risk.'},
  {q:'How can one manage the symptoms of Alzheimer’s?', keys:['manage','routine'], a:'A qualified care team can help develop a plan for symptoms, daily routines, and caregiver support. This FAQ provides general education and cannot prescribe an individual care plan.'},
  {q:'Which models does this project use?', keys:['model','cnn','vgg','accuracy','project'], a:'The supplied MRI checkpoint is a custom CNN with 4,152,356 parameters. The clinical checkpoint is LogisticRegressionCV with 22 named inputs. Slides report a best MRI trial of 91.54%, but that result has not been reproduced with this saved checkpoint. Training accuracy is not documented in the supplied materials.'}
];
function addMessage(role, text) {
  const message = document.createElement('div'); message.className = `message ${role}`;
  const label = document.createElement('strong'); label.textContent = role === 'user' ? 'You' : 'Offline FAQ';
  message.append(label, document.createTextNode(text)); $('#messages').append(message);
}
function answer(prompt) {
  const normalized = prompt.toLowerCase();
  const exact = FAQ.find(item => item.q.toLowerCase() === normalized);
  const found = exact || FAQ.map(item => ({item, score:item.keys.filter(key => normalized.includes(key)).length})).sort((a,b) => b.score - a.score)[0];
  return exact ? exact.a : found?.score > 0 ? found.item.a : 'I can show fixed FAQ answers about Alzheimer’s and this project. Please choose one of the suggestions. This static edition cannot generate Gemini responses.';
}
FAQ.slice(0,10).forEach(item => {const button = document.createElement('button'); button.textContent = item.q; button.addEventListener('click', () => {$('#prompt').value = item.q; $('#prompt').focus();}); $('#suggestions').append(button);});
$('#chat-form').addEventListener('submit', event => {
  event.preventDefault(); const prompt = $('#prompt').value.trim(); if (!prompt) return;
  addMessage('user', prompt); addMessage('assistant', answer(prompt)); $('#prompt').value = '';
  if ($('#messages').children.length > 40) {$('#messages').firstElementChild.remove(); $('#messages').firstElementChild.remove();}
});
$('#new-chat').addEventListener('click', () => $('#messages').replaceChildren());

let rows = [];
const categories = ['AD','LMCI','CN'], colors = {AD:'#d26a3e', LMCI:'#268d87', CN:'#a2c5ac'};
const numeric = ['Age','Education_Year','MMSE'];
const layout = title => ({title:{text:title,font:{size:17}},paper_bgcolor:'#0c1b33',plot_bgcolor:'#0c1b33',font:{color:'#cdd7e7'},margin:{l:50,r:20,t:65,b:70},legend:{orientation:'h'},autosize:true});
const plotConfig = {responsive:true,displayModeBar:false};
function plot(id, traces, title, extra={}) {return Plotly.newPlot(id, traces, {...layout(title),...extra}, plotConfig);}
function redrawCharts() {
  if (!rows.length || !window.Plotly) return;
  const field = $('#category-feature').value;
  const groups = [...new Set(rows.map(row => String(row[field])))].sort();
  const totals = new Map(groups.map(group => [group, rows.filter(row => String(row[field]) === group)]));
  plot('category-chart', categories.map(category => ({type:'bar', name:category, x:groups, y:groups.map(group => {const subset = totals.get(group);return 100 * subset.filter(row => row['DX.bl'] === category).length / subset.length;}),marker:{color:colors[category]}})), `Percent Alzheimer Category By ${field}`, {barmode:'group',yaxis:{title:'Percent'}});
  const x = $('#scatter-x').value, y = $('#scatter-y').value;
  const sampled = rows.filter((_, i) => i % Math.max(1, Math.ceil(rows.length / 10000)) === 0);
  plot('scatter-chart', categories.map(category => {const subset = sampled.filter(row => row['DX.bl'] === category);return {type:'scatter',mode:'markers',name:category,x:subset.map(row => row[x]),y:subset.map(row => row[y]),marker:{color:colors[category],size:5,opacity:.6}};}), `${y} vs ${x}`, {xaxis:{title:x},yaxis:{title:y}});
  const hist = $('#hist-feature').value;
  plot('hist-chart', categories.map(category => ({type:'histogram', name:category, x:rows.filter(row => row['DX.bl'] === category).map(row => row[hist]),nbinsx:30,opacity:.65,marker:{color:colors[category]}})), `Histogram of ${hist}`, {barmode:'overlay'});
  function correlation(a,b) {
    const valid = rows.filter(row => Number.isFinite(row[a]) && Number.isFinite(row[b]));
    if (valid.length < 2) return null;
    const meanA = valid.reduce((sum,row) => sum + row[a], 0) / valid.length;
    const meanB = valid.reduce((sum,row) => sum + row[b], 0) / valid.length;
    let covariance=0, varianceA=0, varianceB=0;
    valid.forEach(row => {const da=row[a]-meanA, db=row[b]-meanB;covariance+=da*db;varianceA+=da*da;varianceB+=db*db;});
    return varianceA && varianceB ? covariance / Math.sqrt(varianceA * varianceB) : null;
  }
  plot('corr-chart', [{type:'heatmap',x:numeric,y:numeric,z:numeric.map(a => numeric.map(b => correlation(a,b))),colorscale:'RdBu',zmin:-1,zmax:1}], 'Correlation Matrix');
  const number = $('#strip-number').value, categoryField = $('#strip-category').value;
  plot('strip-chart', categories.map(category => {const subset = sampled.filter(row => row['DX.bl'] === category);return {type:'scatter',mode:'markers',name:category,x:subset.map(row => row[number]),y:subset.map(row => String(row[categoryField])),marker:{color:colors[category],size:5,opacity:.55}};}), `${number} with ${categoryField}`, {xaxis:{title:number},yaxis:{title:categoryField}});
}
$('#csv-file').addEventListener('change', async event => {
  const file = event.target.files[0]; if (!file) return;
  rows = []; $('#dataset-charts').hidden = true;
  if (file.size > 20 * 1024 * 1024) {$('#csv-status').textContent = 'Choose a CSV smaller than 20 MB.';return;}
  $('#csv-status').textContent = 'Reading your CSV locally…';
  try {
    await Promise.all([loadScript('vendor/papaparse.min.js'),loadScript('vendor/plotly.min.js')]);
    const data = Papa.parse(await file.text(), {header:true,dynamicTyping:true,skipEmptyLines:true,transformHeader:header => header.replace(/^\uFEFF/, '').trim()});
    if (data.errors.length) throw new Error('CSV formatting errors were found. Check quotes and column counts.');
    const required = [...numeric,'Gender','Ethnicity','APOE4_Status','Imputed_Genotype','APOE_Genotype','DX.bl'];
    const missing = required.filter(name => !data.meta.fields.includes(name));
    if (missing.length) throw new Error(`Missing columns: ${missing.join(', ')}.`);
    rows = data.data.filter(row => categories.includes(row['DX.bl']));
    if (!rows.length) throw new Error('No AD, CN or LMCI records found in DX.bl.');
    if (rows.length > 100000) throw new Error('Use a CSV with no more than 100,000 records.');
    numeric.forEach(name => rows.forEach(row => {if (row[name] !== null && row[name] !== '') {const value = Number(row[name]);row[name] = Number.isFinite(value) ? value : null;}}));
    $('#csv-status').textContent = `${rows.length.toLocaleString()} labelled records loaded locally. Scatter plots display at most 10,000 evenly sampled records; counts and correlations use all loaded records.`;
    $('#dataset-charts').hidden = false; redrawCharts();
  } catch(error) {rows=[];$('#csv-status').textContent=error.message;}
});
$('#dataset-charts').querySelectorAll('select').forEach(select => select.addEventListener('change',redrawCharts));
$('#dataset-charts').querySelectorAll('details').forEach(details => details.addEventListener('toggle',() => {
  if (details.open && window.Plotly) details.querySelectorAll('.chart').forEach(chart => Plotly.Plots.resize(chart));
}));
const chartObserver = new ResizeObserver(entries => {
  if (window.Plotly) entries.forEach(({target}) => {
    if (target.offsetWidth > 0 && target.data) requestAnimationFrame(() => Plotly.Plots.resize(target));
  });
});
document.querySelectorAll('.chart').forEach(chart => chartObserver.observe(chart));
navigate(location.hash.slice(1));
