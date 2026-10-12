// Development validation only. Starts its own local file server.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {chromium} = require('playwright');
const root = path.resolve(__dirname, '..');
(async () => {
  const http = require('node:http');
  const mime = {'.html':'text/html','.js':'text/javascript','.mjs':'text/javascript','.css':'text/css','.json':'application/json','.wasm':'application/wasm','.svg':'image/svg+xml','.png':'image/png'};
  const server = http.createServer((request,response) => {
    const pathname = decodeURIComponent(new URL(request.url,'http://localhost').pathname);
    const file = path.resolve(root,'.'+(pathname==='/'?'/index.html':pathname));
    if (!file.startsWith(root+path.sep) || !fs.existsSync(file) || !fs.statSync(file).isFile()) {response.writeHead(404);response.end();return;}
    response.writeHead(200,{'Content-Type':mime[path.extname(file)]||'application/octet-stream'});fs.createReadStream(file).pipe(response);
  });
  await new Promise(resolve=>server.listen(8765,'127.0.0.1',resolve));
  server.unref();
  const options = {headless:true};
  if (process.env.CHROMIUM_PATH) options.executablePath = process.env.CHROMIUM_PATH;
  const browser = await chromium.launch(options);
  const page = await browser.newPage({viewport:{width:1440,height:1000}});
  const errors = []; const external = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('request', request => {if (!request.url().startsWith('http://127.0.0.1:8765') && !request.url().startsWith('blob:') && !request.url().startsWith('data:')) external.push(request.url());});
  await page.goto('http://127.0.0.1:8765/');
  await page.waitForSelector('#home-animation svg');
  await page.screenshot({path:path.join(root,'validation/home-desktop.png'),fullPage:true});
  await page.getByRole('tab',{name:'Aim',exact:true}).click();
  assert.equal(await page.locator('#aim').isVisible(),true);
  await page.selectOption('#page-picker','clinical');
  await page.click('#clinical-predict'); await page.waitForSelector('#clinical-result:not([hidden])');
  assert.match(await page.locator('#clinical-result').innerText(), /AD|CN|LMCI/);
  // Check browser JS math against the exported sklearn fixtures.
  const clinicalDifference = await page.evaluate(async () => {
    const {clinicalProbabilities} = await import('./models.js');
    const model = await (await fetch('models/clinical.json')).json();
    const fixture = await (await fetch('validation/clinical-fixtures.json')).json();
    return Math.max(...fixture.features.flatMap((vector,i) => clinicalProbabilities(model,vector).map((score,j) => Math.abs(score-fixture.expected[i][j]))));
  });
  assert.ok(clinicalDifference < 1e-12);
  const csv = 'Age,Education_Year,MMSE,Gender,Ethnicity,APOE4_Status,Imputed_Genotype,APOE_Genotype,DX.bl\n65,12,20,Male,Unknown,0,True,"(2,2)",AD\n70,14,24,Female,Unknown,1,False,"(3,4)",LMCI\n60,16,29,Male,Unknown,0,False,"(3,3)",CN\n';
  await page.click('#dataset-details summary');
  await page.setInputFiles('#csv-file',{name:'fixture.csv',mimeType:'text/csv',buffer:Buffer.from(csv)});
  await page.waitForSelector('#category-chart .main-svg');
  await page.waitForFunction(() => document.querySelector('#csv-status').textContent.includes('3 labelled'));
  await page.selectOption('#category-feature','APOE_Genotype');
  await page.selectOption('#page-picker','chatbot');
  await page.locator('#suggestions button').first().click(); await page.click('#chat-form button');
  assert.equal(await page.locator('#messages .message').count(),2);
  await page.fill('#prompt','<img src=x onerror=alert(1)>'); await page.click('#chat-form button');
  assert.equal(await page.locator('#messages img').count(),0);
  await page.click('#new-chat');assert.equal(await page.locator('#messages .message').count(),0);
  await page.selectOption('#page-picker','about'); await page.waitForSelector('#about-animation svg');
  assert.equal(await page.locator('.member').count(),4);
  await page.selectOption('#page-picker','mri');
  await page.setInputFiles('#mri-file',path.join(root,'validation/sample-mri.png'));
  await page.waitForSelector('#mri-result:not([hidden])',{timeout:120000});
  assert.match(await page.locator('#mri-status').innerText(),/Prediction complete/);
  // Direct tensor check isolates conversion accuracy from image resize differences.
  const mriDifference = await page.evaluate(async () => {
    const session = await ort.InferenceSession.create('models/mri.onnx',{executionProviders:['wasm']});
    const data = new Float32Array(await (await fetch('validation/mri-input.f32')).arrayBuffer());
    const expected = (await (await fetch('validation/mri-expected.json')).json())[0];
    const results = await session.run({[session.inputNames[0]]:new ort.Tensor('float32',data,[1,176,208,3])});
    const actual = Array.from(results[session.outputNames[0]].data);
    await session.release();
    return {actual,expected,maxDifference:Math.max(...actual.map((value,i)=>Math.abs(value-expected[i])))};
  });
  assert.ok(mriDifference.maxDifference < 1e-4,JSON.stringify(mriDifference));
  await page.screenshot({path:path.join(root,'validation/mri-desktop.png'),fullPage:true});
  await page.setInputFiles('#mri-file',{name:'bad.txt',mimeType:'text/plain',buffer:Buffer.from('bad')});
  assert.match(await page.locator('#mri-status').innerText(),/Choose a JPG/);
  const overflow = [];
  for (const width of [1440,1024,768,390,320]) {
    await page.setViewportSize({width,height:900});
    for (const id of ['home','clinical','mri','chatbot','about']) {
      await page.evaluate(id => {location.hash=id;},id);
      await page.waitForFunction(id => !document.getElementById(id).hidden,id);
      const dimensions = await page.evaluate(() => ({scroll:document.documentElement.scrollWidth,view:innerWidth}));
      if (dimensions.scroll > dimensions.view) overflow.push({id,width,...dimensions});
    }
  }
  await page.setViewportSize({width:390,height:844});
  await page.evaluate(()=>{location.hash='home';});await page.waitForFunction(()=>!document.querySelector('#home').hidden);
  await page.getByRole('tab',{name:'Definition',exact:true}).click();
  await page.screenshot({path:path.join(root,'validation/home-mobile.png'),fullPage:true,animations:'disabled'});
  await page.click('#menu-toggle'); await page.selectOption('#page-picker','clinical');
  assert.equal(await page.locator('#clinical').isVisible(),true);
  assert.equal(await page.locator('body').evaluate(body=>body.classList.contains('nav-open')),false);
  assert.deepEqual(overflow,[]);assert.deepEqual(errors,[]);assert.deepEqual(external,[]);
  const report={browser:'Chromium',clinicalMaxDifference:clinicalDifference,mri:mriDifference,pagesChecked:5,viewportWidths:[1440,1024,768,390,320],horizontalOverflow:overflow,pageErrors:errors,externalRequests:external,checks:['Tabs','Clinical prediction','CSV charts','Offline FAQ','Escaped chat input','MRI upload and inference','Invalid MRI file','Mobile navigation'],publicHosting:'Not deployed or tested on public hosts'};
  fs.writeFileSync(path.join(root,'validation/browser.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));
  await browser.close();
  server.close();
})().catch(error=>{console.error(error);process.exit(1)});
