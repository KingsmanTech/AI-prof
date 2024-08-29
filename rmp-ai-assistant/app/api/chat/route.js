import { NextResponse } from 'next/server'
import { Pinecone } from '@pinecone-database/pinecone'


const PINECONE_API_KEY = "Your Api Key";
const HUGGINGFACE_API_KEY = "Your Api Key";

const systemPrompt = `
You are a rate my professor agent to help students find classes, that takes in user questions and answers them.
For every user question, the top 3 professors that match the user question are returned.
Use them to answer the question if needed.
`

async function getEmbedding(text) {
    const response = await fetch(
        "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2",
        {
            headers: { 
                Authorization: `Bearer ${HUGGINGFACE_API_KEY}`,
                "Content-Type": "application/json"
            },
            method: "POST",
            body: JSON.stringify({ inputs: text }),
        }
    );
    if (!response.ok) {
        const errorBody = await response.text();
        console.error(`HTTP error! status: ${response.status}, body: ${errorBody}`);
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    const result = await response.json();
    if (Array.isArray(result) && result.length > 0 && Array.isArray(result[0])) {
        return result[0];  // Return the first (and only) embedding
    } else {
        console.error("Unexpected response format:", result);
        throw new Error("Unexpected response format from Hugging Face API");
    }
}

async function getCompletion(messages) {
    const response = await fetch(
        "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill",
        {
            headers: { 
                Authorization: `Bearer ${HUGGINGFACE_API_KEY}`,
                "Content-Type": "application/json"
            },
            method: "POST",
            body: JSON.stringify({ inputs: messages.map(m => m.content).join('\n') }),
        }
    );
    if (!response.ok) {
        const errorBody = await response.text();
        console.error(`HTTP error! status: ${response.status}, body: ${errorBody}`);
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    const result = await response.json();
    if (Array.isArray(result) && result.length > 0 && result[0].generated_text) {
        return result[0].generated_text;
    } else {
        console.error("Unexpected response format:", result);
        throw new Error("Unexpected response format from Hugging Face API");
    }
}

export async function POST(req) {
    try {
        const data = await req.json()
        
        const pc = new Pinecone({
            apiKey: PINECONE_API_KEY,
        })
        const index = pc.Index('rag').namespace('ns1')

        const text = data[data.length - 1].content
        console.log("Requesting embedding for text:", text);
        const embedding = await getEmbedding(text)
        console.log("Received embedding:", embedding);

        const results = await index.query({
            topK: 3,
            includeMetadata: true,
            vector: embedding,
        })

        let resultString = ''
        results.matches.forEach((match) => {
            resultString += `
            Returned Results:
            Professor: ${match.id}
            Review: ${match.metadata.review}
            Subject: ${match.metadata.subject}
            Stars: ${match.metadata.stars}
            \n\n`
        })

        const lastMessage = data[data.length - 1]
        const lastMessageContent = lastMessage.content + resultString
        const lastDataWithoutLastMessage = data.slice(0, data.length - 1)

        const completion = await getCompletion([
            {role: 'system', content: systemPrompt},
            ...lastDataWithoutLastMessage,
            {role: 'user', content: lastMessageContent},
        ])

        return NextResponse.json({ response: completion })
    } catch (error) {
        console.error("Error in POST request:", error)
        return NextResponse.json({ error: "An error occurred while processing your request" }, { status: 500 })
    }
}