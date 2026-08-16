import { createWorkersAI } from "workers-ai-provider";
import { callable, routeAgentRequest, type Schedule, Agent } from "agents";
import { getSchedulePrompt, scheduleSchema } from "agents/schedule";
import { AIChatAgent, type OnChatMessageOptions } from "@cloudflare/ai-chat";
import {
  withVoice,
  WorkersAIFluxSTT,
  WorkersAITTS,
  type VoiceTurnContext
} from "@cloudflare/voice";
import {
  convertToModelMessages,
  pruneMessages,
  stepCountIs,
  streamText,
  tool
} from "ai";
import { z } from "zod";


const VoiceAgent = withVoice(AIChatAgent);

export class ChatAgent extends VoiceAgent<Env> {
  maxPersistedMessages = 100;
  chatRecovery = true;
  // Wait for MCP connections to be re-established after hibernation before
  // processing a message, so MCP tools aren't intermittently missing.
  waitForMcpConnections = true;

  transcriber = new WorkersAIFluxSTT(this.env.AI);
  tts = new WorkersAITTS(this.env.AI);

  async onTurn(transcript: string, context: VoiceTurnContext) {
    // Forward voice transcript to the existing chat message handler
    const response = await this.onChatMessage(null, {
      requestId: crypto.randomUUID(),
    });

    // Since onChatMessage returns a stream, we need to handle how it
    // integrates with the voice pipeline's expected string/stream return.
    // For now, we'll implement a basic response or you can customize the logic.
    return `You said: ${transcript}`;
  }

  onStart() {
    // Configure OAuth popup behavior for MCP servers that require authentication
    this.mcp.configureOAuthCallback({
      customHandler: (result) => {
        if (result.authSuccess) {
          return new Response("<script>window.close();</script>", {
            headers: { "content-type": "text/html" },
            status: 200
          });
        }
        return new Response(
          `Authentication Failed: ${result.authError || "Unknown error"}`,
          { headers: { "content-type": "text/plain" }, status: 400 }
        );
      }
    });
  }


  @callable()
  async addServer(name: string, url: string) {
    return await this.addMcpServer(name, url);
  }


  @callable()
  async removeServer(serverId: string) {
    await this.removeMcpServer(serverId);
  }


  async onChatMessage(_onFinish: unknown, options?: OnChatMessageOptions) {
    const mcpTools = this.mcp.getAITools();
    const workersai = createWorkersAI({ binding: this.env.AI });


    const result = streamText({
      model: workersai("@cf/google/gemma-4-26b-a4b-it", {
        sessionAffinity: this.sessionAffinity
      }),
      system: `You are a helpful assistant that can understand images, run calculations, schedule tasks, and access files stored across connected R2 buckets.


You have access to three R2 storage buckets:
- shopify-skill bucket (via listR2Files to browse, getR2File to read)
- agentic-commerce bucket (via listAgenticCommerceFiles to browse, getAgenticCommerceFile to read)
- cloudflare-skills bucket (via listCloudflareSkillFiles to browse, getCloudflareSkillFile to read)


When asked about skills, Shopify data, agentic commerce, Cloudflare configurations, or files stored in these buckets, use the appropriate tool to fetch the file contents.
Besides the R2 buckets, you also have access to the connected mcp servers and their tools. If a tool is not available, you can ask the user to connect to the server or provide the necessary information.
All executed codes must be and has to always be Production ready code, the less the dependencies the better it is during the execution of the code. If you are not sure about the code, ask the user for clarification or more information.

Your Name is Genesis Your The King at what you do and you have fun and Love doing what you do including helping the user.

THE NUMBER 1 RULE IS: RESPECT THE CODE RESPECT THE WORKSPACE RESPECT THE USER AND RESPECT YOURSELF AS A GENIUS.

${getSchedulePrompt({ date: new Date() })}


If the user asks to schedule a task, use the schedule tool to schedule the task.`,
      // Prune old tool calls and reasoning to save tokens on long conversations
      messages: pruneMessages({
        messages: await convertToModelMessages(this.messages),
        toolCalls: "before-last-2-messages",
        reasoning: "before-last-message"
      }),
      tools: {
        // MCP tools from connected servers
        ...mcpTools,


      getR2File: tool({
          description:
            "Read any file from the shopify-skill R2 bucket.",
          inputSchema: z.object({
            key: z.string().describe("Exact R2 object key")
          }),
          execute: async ({ key }) => {
            const object = await this.env.R2.get(key);

            return {
              key,
              contentType: object?.httpMetadata?.contentType ?? "unknown",
              content: await object?.text()
            };
          }
        }),

        listR2Files: tool({
          description:
            "List files and folders in the shopify-skill R2 bucket. Use prefix to browse into a folder (e.g. 'folder/'), delimiter '/' to group into folders, cursor for pagination.",
          inputSchema: z.object({
            prefix: z.string().optional().describe("Folder prefix to list (e.g. 'folder/'). Omit to list root."),
            delimiter: z.string().optional().describe("Delimiter to group keys into folders (use '/')."),
            cursor: z.string().optional().describe("Pagination cursor from a previous truncated result."),
            limit: z.number().optional().describe("Max objects per page (default 1000, max 1000).")
          }),
          execute: async ({ prefix, delimiter, cursor, limit }) => {
            const result = await this.env.R2.list({
              prefix: prefix || undefined,
              delimiter: delimiter || undefined,
              cursor: cursor || undefined,
              limit: limit || 1000
            });
            return {
              folders: result.delimitedPrefixes,
              files: result.objects.map(o => ({
                key: o.key,
                size: o.size,
                contentType: o.httpMetadata?.contentType ?? "unknown"
              })),
              truncated: result.truncated,
              cursor: result.truncated ? result.cursor : null
            };
          }
        }),

                getAgenticCommerceFile: tool({
          description:
            "Read any file from the agentic-commerce R2 bucket.",
          inputSchema: z.object({
            key: z.string().describe("Exact R2 object key")
          }),
          execute: async ({ key }) => {
            const object = await this.env["r2-agentic-commerce"].get(key);

            return {
              key,
              contentType: object?.httpMetadata?.contentType ?? "unknown",
              content: await object?.text()
            };
          }
        }),

        listAgenticCommerceFiles: tool({
          description:
            "List files and folders in the agentic-commerce R2 bucket. Use prefix to browse into a folder (e.g. 'folder/'), delimiter '/' to group into folders, cursor for pagination.",
          inputSchema: z.object({
            prefix: z.string().optional().describe("Folder prefix to list (e.g. 'folder/'). Omit to list root."),
            delimiter: z.string().optional().describe("Delimiter to group keys into folders (use '/')."),
            cursor: z.string().optional().describe("Pagination cursor from a previous truncated result."),
            limit: z.number().optional().describe("Max objects per page (default 1000, max 1000).")
          }),
          execute: async ({ prefix, delimiter, cursor, limit }) => {
            const result = await this.env["r2-agentic-commerce"].list({
              prefix: prefix || undefined,
              delimiter: delimiter || undefined,
              cursor: cursor || undefined,
              limit: limit || 1000
            });
            return {
              folders: result.delimitedPrefixes,
              files: result.objects.map(o => ({
                key: o.key,
                size: o.size,
                contentType: o.httpMetadata?.contentType ?? "unknown"
              })),
              truncated: result.truncated,
              cursor: result.truncated ? result.cursor : null
            };
          }
        }),

                getCloudflareSkillFile: tool({
          description:
            "Read any file from the cloudflare-skills R2 bucket.",
          inputSchema: z.object({
            key: z.string().describe("Exact R2 object key")
          }),
          execute: async ({ key }) => {
            const object = await this.env["r2-cloudflare"].get(key);

            return {
              key,
              contentType: object?.httpMetadata?.contentType ?? "unknown",
              content: await object?.text()
            };
          }
        }),

        listCloudflareSkillFiles: tool({
          description:
            "List files and folders in the cloudflare-skills R2 bucket. Use prefix to browse into a folder (e.g. 'folder/'), delimiter '/' to group into folders, cursor for pagination.",
          inputSchema: z.object({
            prefix: z.string().optional().describe("Folder prefix to list (e.g. 'folder/'). Omit to list root."),
            delimiter: z.string().optional().describe("Delimiter to group keys into folders (use '/')."),
            cursor: z.string().optional().describe("Pagination cursor from a previous truncated result."),
            limit: z.number().optional().describe("Max objects per page (default 1000, max 1000).")
          }),
          execute: async ({ prefix, delimiter, cursor, limit }) => {
            const result = await this.env["r2-cloudflare"].list({
              prefix: prefix || undefined,
              delimiter: delimiter || undefined,
              cursor: cursor || undefined,
              limit: limit || 1000
            });
            return {
              folders: result.delimitedPrefixes,
              files: result.objects.map(o => ({
                key: o.key,
                size: o.size,
                contentType: o.httpMetadata?.contentType ?? "unknown"
              })),
              truncated: result.truncated,
              cursor: result.truncated ? result.cursor : null
            };
          }
        }),

       
        // Client-side tool: no execute function — the browser handles it
        getUserTimezone: tool({
          description:
            "Get the user's timezone from their browser. Use this when you need to know the user's local time.",
          inputSchema: z.object({})
        }),


        // Approval tool: requires user confirmation before executing
        calculate: tool({
          description:
            "Perform a math calculation with two numbers. Requires user approval for large numbers.",
          inputSchema: z.object({
            a: z.number().describe("First number"),
            b: z.number().describe("Second number"),
            operator: z
              .enum(["+", "-", "*", "/", "%"])
              .describe("Arithmetic operator")
          }),
          needsApproval: async ({ a, b }) =>
            Math.abs(a) > 1000 || Math.abs(b) > 1000,
          execute: async ({ a, b, operator }) => {
            const ops: Record<string, (x: number, y: number) => number> = {
              "+": (x, y) => x + y,
              "-": (x, y) => x - y,
              "*": (x, y) => x * y,
              "/": (x, y) => x / y,
              "%": (x, y) => x % y
            };
            if (operator === "/" && b === 0) {
              return { error: "Division by zero" };
            }
            return {
              expression: `${a} ${operator} ${b}`,
              result: ops[operator](a, b)
            };
          }
        }),


        scheduleTask: tool({
          description:
            "Schedule a task to be executed at a later time. Use this when the user asks to be reminded or wants something done later.",
          inputSchema: scheduleSchema,
          execute: async ({ when, description }) => {
            if (when.type === "no-schedule") {
              return "Not a valid schedule input";
            }
            const input =
              when.type === "scheduled"
                ? when.date
                : when.type === "delayed"
                  ? when.delayInSeconds
                  : when.type === "cron"
                    ? when.cron
                    : null;
            if (!input) return "Invalid schedule type";
            try {
              this.schedule(input, "executeTask", description, {
                idempotent: true
              });
              return `Task scheduled: "${description}" (${when.type}: ${input})`;
            } catch (error) {
              return `Error scheduling task: ${error}`;
            }
          }
        }),


        getScheduledTasks: tool({
          description: "List all tasks that have been scheduled",
          inputSchema: z.object({}),
          execute: async () => {
            const tasks = this.getSchedules();
            return tasks.length > 0 ? tasks : "No scheduled tasks found.";
          }
        }),


        cancelScheduledTask: tool({
          description: "Cancel a scheduled task by its ID",
          inputSchema: z.object({
            taskId: z.string().describe("The ID of the task to cancel")
          }),
          execute: async ({ taskId }) => {
            try {
              this.cancelSchedule(taskId);
              return `Task ${taskId} cancelled.`;
            } catch (error) {
              return `Error cancelling task: ${error}`;
            }
          }
        })
      },
      stopWhen: stepCountIs(20),
      abortSignal: options?.abortSignal
    });


    return result.toUIMessageStreamResponse();
  }


  async executeTask(description: string, _task: Schedule<string>) {
    // Do the actual work here (send email, call API, etc.)
    console.log(`Executing scheduled task: ${description}`);


    // Notify connected clients via a broadcast event.
    // We use broadcast() instead of saveMessages() to avoid injecting
    // into chat history — that would cause the AI to see the notification
    // as new context and potentially loop.
    this.broadcast(
      JSON.stringify({
        type: "scheduled-task",
        description,
        timestamp: new Date().toISOString()
      })
    );
  }
}


export default {
  async fetch(request: Request, env: Env) {
    return (
      (await routeAgentRequest(request, env)) ||
      new Response("Not found", { status: 404 })
    );
  }
} satisfies ExportedHandler<Env>;