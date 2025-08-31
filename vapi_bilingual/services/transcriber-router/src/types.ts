export type VapiStart = {
 type: 'start';
 encoding: 'linear16';
 container: 'raw';
 sampleRate: number; // 16000
 channels: number; // 2
};


export type VapiTranscriberResponse = {
 type: 'transcriber-response';
 transcription: string;
 channel: 'customer' | 'assistant';
};