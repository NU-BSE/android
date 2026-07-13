export type Recording = {
  id: string;
  game_code: string;
  file_name: string;
  sha256: string;
  byte_size: number;
  record_count: number;
  transition_count: number;
  game_id: string | null;
  frame_shapes: Record<string, number>;
  available_actions: number[];
  download_path: string;
  signed_url_path: string;
};

export class ArcDatasetApi {
  constructor(
    private readonly baseUrl: string,
    private readonly apiKey?: string,
  ) {}

  private headers(extra?: HeadersInit): Headers {
    const headers = new Headers(extra);
    if (this.apiKey) headers.set("X-API-Key", this.apiKey);
    return headers;
  }

  private url(path: string): URL {
    // Preserve reverse-proxy prefixes such as /agi-3. A leading slash passed to
    // new URL(path, baseUrl) would otherwise resolve against the site root.
    const base = this.baseUrl.replace(/\/+$/, "");
    const suffix = path.replace(/^\/+/, "");
    return new URL(`${base}/${suffix}`);
  }

  async listRecordings(gameCode?: string): Promise<Recording[]> {
    const all: Recording[] = [];
    let offset = 0;

    while (true) {
      const url = this.url("v1/recordings");
      url.searchParams.set("limit", "500");
      url.searchParams.set("offset", String(offset));
      if (gameCode) url.searchParams.set("game_code", gameCode);

      const response = await fetch(url, { headers: this.headers() });
      if (!response.ok) throw new Error(`Dataset API returned ${response.status}`);
      const page = (await response.json()) as { items: Recording[]; total: number };
      all.push(...page.items);
      offset += page.items.length;
      if (offset >= page.total || page.items.length === 0) return all;
    }
  }

  async getRecording(recordingId: string): Promise<Recording> {
    const url = this.url(`v1/recordings/${encodeURIComponent(recordingId)}`);
    const response = await fetch(url, { headers: this.headers() });
    if (!response.ok) throw new Error(`Recording request returned ${response.status}`);
    return (await response.json()) as Recording;
  }

  async *streamRecording(recordingId: string): AsyncGenerator<unknown> {
    const url = this.url(
      `v1/recordings/${encodeURIComponent(recordingId)}/download`,
    );
    const response = await fetch(url, { headers: this.headers() });
    if (!response.ok || !response.body) {
      throw new Error(`Dataset stream returned ${response.status}`);
    }

    const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
    let buffer = "";
    try {
      while (true) {
        const { value, done } = await reader.read();
        buffer += value ?? "";
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        for (const line of lines) {
          if (line.trim()) yield JSON.parse(line);
        }
        if (done) break;
      }
      if (buffer.trim()) yield JSON.parse(buffer);
    } finally {
      reader.releaseLock();
    }
  }

  async signedDownloadUrl(recordingId: string): Promise<string> {
    const url = this.url(
      `v1/recordings/${encodeURIComponent(recordingId)}/signed-url`,
    );
    const response = await fetch(url, { headers: this.headers() });
    if (!response.ok) throw new Error(`Signed URL request returned ${response.status}`);
    return (await response.json()).url as string;
  }
}
