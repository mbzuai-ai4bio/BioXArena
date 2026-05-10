/**
 * Feishu types — standalone definitions without Zod dependency.
 * Replaces the original types.ts which used z.infer<> on Zod schemas.
 */

import type { BaseProbeResult } from "openclaw/plugin-sdk/feishu";

/** MentionTarget — inlined from mention.ts (not copied) */
export type MentionTarget = {
  openId: string;
  name: string;
  key: string;
};

/** Feishu top-level config (simplified — matches runtime shape) */
export type FeishuConfig = {
  enabled?: boolean;
  defaultAccount?: string;
  appId?: string;
  appSecret?: string;
  encryptKey?: string;
  verificationToken?: string;
  domain?: string;
  connectionMode?: string;
  webhookPath?: string;
  webhookHost?: string;
  webhookPort?: number;
  dmPolicy?: string;
  groupPolicy?: string;
  allowFrom?: Array<string | number>;
  groupAllowFrom?: Array<string | number>;
  groupSenderAllowFrom?: Array<string | number>;
  requireMention?: boolean;
  groups?: Record<string, FeishuGroupConfig | undefined>;
  historyLimit?: number;
  dmHistoryLimit?: number;
  textChunkLimit?: number;
  mediaMaxMb?: number;
  httpTimeoutMs?: number;
  tools?: FeishuToolsConfig;
  accounts?: Record<string, FeishuAccountConfig | undefined>;
  dynamicAgentCreation?: DynamicAgentCreationConfig;
  typingIndicator?: boolean;
  resolveSenderNames?: boolean;
  [key: string]: unknown;
};

export type FeishuGroupConfig = {
  requireMention?: boolean;
  enabled?: boolean;
  allowFrom?: Array<string | number>;
  systemPrompt?: string;
  [key: string]: unknown;
};

export type FeishuAccountConfig = {
  enabled?: boolean;
  name?: string;
  appId?: string;
  appSecret?: string;
  encryptKey?: string;
  verificationToken?: string;
  domain?: string;
  connectionMode?: string;
  webhookPath?: string;
  tools?: FeishuToolsConfig;
  mediaMaxMb?: number;
  httpTimeoutMs?: number;
  [key: string]: unknown;
};

export type FeishuDomain = "feishu" | "lark" | (string & {});
export type FeishuConnectionMode = "websocket" | "webhook";

export type FeishuDefaultAccountSelectionSource =
  | "explicit-default"
  | "mapped-default"
  | "fallback";
export type FeishuAccountSelectionSource = "explicit" | FeishuDefaultAccountSelectionSource;

export type ResolvedFeishuAccount = {
  accountId: string;
  selectionSource: FeishuAccountSelectionSource;
  enabled: boolean;
  configured: boolean;
  name?: string;
  appId?: string;
  appSecret?: string;
  encryptKey?: string;
  verificationToken?: string;
  domain: FeishuDomain;
  config: FeishuConfig;
};

export type FeishuIdType = "open_id" | "user_id" | "union_id" | "chat_id";

export type FeishuMessageContext = {
  chatId: string;
  messageId: string;
  senderId: string;
  senderOpenId: string;
  senderName?: string;
  chatType: "p2p" | "group" | "private";
  mentionedBot: boolean;
  hasAnyMention?: boolean;
  rootId?: string;
  parentId?: string;
  threadId?: string;
  content: string;
  contentType: string;
  mentionTargets?: MentionTarget[];
};

export type FeishuSendResult = {
  messageId: string;
  chatId: string;
};

export type FeishuChatType = "p2p" | "group" | "private";

export type FeishuMessageInfo = {
  messageId: string;
  chatId: string;
  chatType?: FeishuChatType;
  senderId?: string;
  senderOpenId?: string;
  senderType?: string;
  content: string;
  contentType: string;
  createTime?: number;
};

export type FeishuProbeResult = BaseProbeResult<string> & {
  appId?: string;
  botName?: string;
  botOpenId?: string;
};

export type FeishuMediaInfo = {
  path: string;
  contentType?: string;
  placeholder: string;
};

export type FeishuToolsConfig = {
  doc?: boolean;
  chat?: boolean;
  wiki?: boolean;
  drive?: boolean;
  perm?: boolean;
  scopes?: boolean;
};

export type DynamicAgentCreationConfig = {
  enabled?: boolean;
  workspaceTemplate?: string;
  agentDirTemplate?: string;
  maxAgents?: number;
};
