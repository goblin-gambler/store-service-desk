import { ScrollArea } from '@/components/ui/scroll-area'
import { Card } from '@/components/ui/card'
import type { Task } from '@/types'

interface TaskDetailProps {
  task: Task | null
}

export function TaskDetail({ task }: TaskDetailProps) {
  if (!task)
    return (
      <div className="bg-white rounded-lg border flex items-center justify-center">
        <div className="text-gray-400">请从左侧选择任务</div>
      </div>
    )

  const snapshot = task.context_snapshot
  const isLossReport = snapshot?.subject_type === 'loss_report'

  return (
    <div className="bg-white rounded-lg border flex flex-col">
      <div className="p-3 border-b">
        <h2 className="font-semibold">任务 #{task.audit_log_id}</h2>
      </div>

      <ScrollArea className="flex-1 p-4">
        <div className="space-y-4">
          <div>
            <h3 className="text-sm font-medium text-gray-500 mb-1">用户问题</h3>
            <Card className="p-3 bg-gray-50">{task.context_snapshot?.question || '无'}</Card>
          </div>

          <div>
            <h3 className="text-sm font-medium text-gray-500 mb-1">触发原因</h3>
            <Card className="p-3 bg-yellow-50 border-yellow-200">{task.trigger_reason}</Card>
          </div>

          {isLossReport && (
            <div>
              <h3 className="text-sm font-medium text-gray-500 mb-1">报损信息</h3>
              <Card className="p-3 border-l-4 border-l-brand-500">
                <div className="flex justify-between">
                  <span className="font-medium">{snapshot?.loss_report_sn}</span>
                  <span className="text-red-500 font-bold">¥{snapshot?.amount}</span>
                </div>
                <div className="text-sm text-gray-600 mt-1">
                  {snapshot?.store_name}（{snapshot?.store_code}）
                </div>
                <div className="text-sm text-gray-600 mt-1">
                  审批层级：{snapshot?.approval_level}
                </div>
              </Card>
            </div>
          )}

          <div>
            <h3 className="text-sm font-medium text-gray-500 mb-1">完整上下文</h3>
            <pre className="text-xs bg-gray-100 p-2 rounded overflow-auto max-h-40">
              {JSON.stringify(task.context_snapshot, null, 2)}
            </pre>
          </div>
        </div>
      </ScrollArea>
    </div>
  )
}
