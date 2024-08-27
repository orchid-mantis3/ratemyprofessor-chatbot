import { NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
const { GoogleGenerativeAI } = require("@google/generative-ai")


const systemPrompt=
`You are an AI assistant specializing in helping students find professors based on their specific needs and preferences. Your knowledge comes from a database of professor reviews and ratings, which you access using a RAG (Retrieval-Augmented Generation) system.

For each user query, you will:

1. Analyze the student's request to understand their requirements.
2. Use the RAG system to retrieve information about the top 3 most relevant professors based on the query.
3. Present these professors to the student in a clear, concise manner.

When presenting each professor, include:
- The professor's name
- Their department or subject area
- A brief summary of their strengths based on student reviews
- Their overall rating (if available)
- Any other relevant information that matches the student's query

After presenting the top 3 professors, offer to provide more details on any of them if the student is interested.

Remember to:
- Be objective and base your recommendations solely on the data provided by the RAG system.
- Avoid making personal judgments or opinions about the professors.
- If the query is too vague or broad, ask for clarification to provide more accurate results.
- If there aren't enough matches in the database, inform the student and suggest broadening their search criteria.

Your goal is to help students make informed decisions about their course selections by providing them with relevant and useful information about professors.
`
export async function POST(req) {
    const data= await req.json()
    const pc=new Pinecone({
        apiKey: process.env.PINECONE_API_KEY,
    })

const index=pc.index('rag').namespace('ns1')
const genAI = new GoogleGenerativeAI(process.env.API_KEY)
const text =data[data.length-1].content
const model= genAI.getGenerativeModel({model:"text-embedding-004"})
const result=await model.embedContent(text)
const embedding= result.embedding
const results= await index.query({
    topK:3,
    includeMetadata:true,
    vector: embedding.values,
    
})
  let resultString='\n\nReturned results from vector db(done automatically):'
  results.matches.forEach((match)=>{
    resultString +=`\n
    Professor:${match.id}
    Review:${match.metadata.stars}
    Subject:${match.metadata.subject}
    Stars ${match.metadata.stars}
    \n\n`
  })
  const model_gen = genAI.getGenerativeModel({ model: 'gemini-1.5-flash' });

  // const completion = await model_gen.generateContentStream(resultString);
  const gen_result = await model_gen.generateContent(`${systemPrompt}\nQuery: ${text}\n${data}\n`);
  const response = await gen_result.response.text();
  const outputWithoutAsterisks = response.replace(/\*/g, ''); 

  return new NextResponse(outputWithoutAsterisks)
}



// export async function POST(request) {
//     try {
//         // Parse the request body
//         const data = await request.json();

//         const pc = new Pinecone({
//             apiKey: process.env.PINECONE_API_KEY
//         });
//         const index = pc.Index('rag');
//         const text = data[data.length - 1].content;

//         // Initialize Gemini AI
//         const genAI = new GoogleGenerativeAI(process.env.API_KEY);

//         // Generate embedding using Gemini
//         const embeddingModel = genAI.getGenerativeModel({ model: "text-embedding-004" });
//         const embeddingResult = await embeddingModel.embedContent(text);
//         const embedding = embeddingResult.embedding;
        
//         const userMsg = data[data.length - 1];

//         const results = await index.query({
//             vector: embedding['values'],
//             topK: 3,
//             includeMetadata: true,
//             // namespace: 'ns1'
//         });

//         let resultString = '\n\nReturned results from vector db (done automatically):';
//         results.matches.forEach((match) => {
//             resultString += `\n
//             Professor: ${match.id}
//             Review: ${match.metadata.review}
//             Subject: ${match.metadata.subject}
//             Stars: ${match.metadata.stars}
//             \n\n
//             `;
//         });

//         const model = genAI.getGenerativeModel({ 
//             model: "gemini-1.5-flash",
//         });

//         async function startChat(data) {
//             return model.startChat({
//                 history: data.map(msg => ({
//                     role: msg.role,
//                     parts: [{ text: msg.content }]
//                 })),
//                 generationConfig: { 
//                     maxOutputTokens: 8000,
//                 },
//             });
//         }

//         const chat = await startChat(data);
//         const result = await chat.sendMessage({
//             contents: [{
//                 role: "user",
//                 parts: [{ text: userMsg.content + resultString }]
//             }]
//         });
//         const response = result.response;
//         const output = response.text();
//         console.log(output);

//         return NextResponse.json({ text: output });
//     } catch (e) {
//         console.error(e);
//         return NextResponse.json({ text: "Error: " + e.message }, { status: 500 });
//     }
// }